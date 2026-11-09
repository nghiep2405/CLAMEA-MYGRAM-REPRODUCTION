assert torch.cuda.is_available(), (
    "Molab GPU chưa bật; dừng tại đây."
)

assert (
    torch.cuda.get_device_properties(0).total_memory
    >= 8 * 1024**3
), "GPU VRAM below 8 GiB"


# CLAMEA upstream có circular import.
# Phải import kgs trước main.
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

import kgs
import main as official

from kgs import KGs
import torch.nn.functional as F


base_config = json.loads(
    (original_source / "args" / "clamea.json").read_text()
)

base_config.update({
    "training_data": str(data_dir) + "/",
    "output": str(WORK / "runs" / "unused") + "/",
    "dataset_division": "721",
    "dataset_division33": "721_5fold/1/",
    "visual_dim": 4096,
    "random_seed": 66,
    "max_epoch": 300,
    "start_valid": 70,
    "eval_freq": 70,
    "iterative": True,
    "early_stop": False,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "dim": 100,
    "batch_size": 5000,
    "e_batch_size": 5000,
    "stop_metric": "mrr",
    "eval_metric": "inner",
    "eval_norm": True,
    "top_k": [1, 5, 10],
    "csls": 10,
    "orth_factor": 0.01,
    "mse_factor": 0.01,
})

assert base_config["dim"] == 100
assert base_config["visual_dim"] == 4096


# Seed before model construction
random.seed(66)
np.random.seed(66)
torch.manual_seed(66)
torch.cuda.manual_seed_all(66)


kg = KGs(
    base_config["training_data"],
    "721",
    66,
    ordered=True,
    modality="ria",
)

args = official.ARGs(base_config)
smoke_model = official.CLAMEA(kg, args).cuda()

assert smoke_model.fc_i.in_features == 4096
assert smoke_model.fc_a.in_features == 768


# Chọn các gold pairs đủ modality cho smoke test
eligible = [
    (e1, e2)
    for e1, e2 in zip(
        kg.train_entities1,
        kg.train_entities2,
    )
    if (
        kg.attr_mask[e1]
        and kg.attr_mask[e2]
        and kg.image_mask[e1]
        and kg.image_mask[e2]
    )
]

assert len(eligible) >= 4, (
    "Insufficient complete-modality gold pairs for smoke"
)

chosen = eligible[:min(8, len(eligible))]


e1 = torch.tensor(
    [p[0] for p in chosen],
    device="cuda",
    dtype=torch.long,
)

e2 = torch.tensor(
    [p[1] for p in chosen],
    device="cuda",
    dtype=torch.long,
)


am1 = torch.tensor(
    [kg.attr_mask[int(x)] for x in e1],
    device="cuda",
    dtype=torch.uint8,
)

am2 = torch.tensor(
    [kg.attr_mask[int(x)] for x in e2],
    device="cuda",
    dtype=torch.uint8,
)

im1 = torch.tensor(
    [kg.image_mask[int(x)] for x in e1],
    device="cuda",
    dtype=torch.uint8,
)

im2 = torch.tensor(
    [kg.image_mask[int(x)] for x in e2],
    device="cuda",
    dtype=torch.uint8,
)


smoke_model.train()

r, a, v, f, orth, vae = smoke_model.predict(
    e1, e2,
    am1, im1,
    am2, im2,
    "train",
)

label = torch.eye(
    len(chosen),
    device="cuda",
)

criterion = official.ContrastiveLoss()

losses = [
    criterion(x, label)
    for x in (f, r, a, v)
]

smoke_loss = sum(losses) + orth + vae

assert all(
    torch.isfinite(x).all()
    for x in losses + [orth, vae, smoke_loss]
), "Non-finite loss"


# Kiểm tra backward
smoke_loss.backward()

grad_count = sum(
    int(
        p.grad is not None
        and torch.isfinite(p.grad).all()
    )
    for p in smoke_model.parameters()
    if p.requires_grad
)

assert grad_count > 20, (
    "Too few finite gradients",
    grad_count,
)


# Independent exact-ranking oracle
S = np.array([
    [0.8, 0.1, 0.2, 0.3],
    [0.3, 0.7, 0.2, 0.1],
    [0.2, 0.1, 0.6, 0.3],
    [0.1, 0.2, 0.3, 0.9],
], dtype=np.float64)

ranks = [
    int(
        np.where(np.argsort(-S[i]) == i)[0][0]
    ) + 1
    for i in range(4)
]

assert ranks == [1, 1, 1, 1]


from evaluation import greedy_alignment

identity = np.eye(4, dtype=np.float64)

_, hits, mr, mrr = greedy_alignment(
    identity,
    identity,
    [1, 5, 10],
    1,
    metric="inner",
    normalize=True,
    csls_k=0,
    accurate=True,
)

assert float(mrr) == 1.0
assert float(hits[0]) == 100.0


SMOKE = {
    "gpu": torch.cuda.get_device_name(0),
    "finite_loss": float(smoke_loss.item()),
    "finite_grad_parameter_tensors": grad_count,
    "gold_complete_pairs": len(chosen),
    "exact_rank_oracle": "PASS",
    "visual_dim": 4096,
    "attr_dim": 768,
}

print("GPU SMOKE PASS:")
print(json.dumps(SMOKE, indent=2))


# Cleanup
del smoke_model, smoke_loss, kg

import gc

gc.collect()
torch.cuda.empty_cache()
