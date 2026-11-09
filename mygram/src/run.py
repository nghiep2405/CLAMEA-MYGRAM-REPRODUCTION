#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import json
import os
import random
import signal
import time
from pprint import pprint
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers import (
    get_cosine_schedule_with_warmup,
)

from .checkpoint import (
    append_jsonl,
    initialize_run_directory,
    load_checkpoint,
    save_checkpoint,
    write_predictions,
)
from .config import emit_resolved, parse_args, resolved_dict
from .data import load_dataset, save_split
from .data_registry import sha256, write_json
from .evaluation import evaluate_embeddings
from .iterative_learning import (
    IterativeLearningState,
    diagnostic_precision,
    mutual_nearest_pairs,
    ordered_il_state,
    update_il_state,
)
from .Load import load_data_dual, load_word_char_features
from .loss import CustomMultiLossLayer, GRAM_Loss, InfoNCE_loss, alignment_objective
from .models import MultiModalEncoderMrFusion


def apply_optimizer_policy(optimizer, params, args, epoch):
    """Apply the selected LR/reset behavior and return (optimizer, event)."""
    if epoch < args.il_start:
        return optimizer, None
    if args.optimizer_reset_policy == 'every_epoch_after_il':
        reset_groups = []
        for group in params:
            if isinstance(group, dict):
                reset_group = dict(group)
                reset_group['lr'] = args.lr / 5
                reset_group['weight_decay'] = args.weight_decay
                reset_groups.append(reset_group)
            else:
                reset_groups.append(group)
        return (
            optim.AdamW(reset_groups, lr=args.lr / 5, weight_decay=args.weight_decay),
            'optimizer_reset',
        )
    if epoch == args.il_start:
        for group in optimizer.param_groups:
            group['lr'] = args.lr / 5
        return optimizer, 'learning_rate_changed'
    return optimizer, None




class MyGram:

    def __init__(self, args=None):

        self.ent2id_dict = None
        self.ills = None
        self.triples = None
        self.r_hs = None
        self.r_ts = None
        self.ids = None
        self.left_ents = None
        self.right_ents = None

        self.img_features = None
        self.rel_features = None
        self.att_features = None
        self.char_features = None
        self.name_features = None
        self.ent_vec = None

        self.left_non_train = None
        self.right_non_train = None
        self.ENT_NUM = None
        self.REL_NUM = None
        self.mcd_operator = None
        self.train_ill = None
        self.test_ill_ = None
        self.test_ill = None
        self.test_left = None
        self.test_right = None

        self.multimodal_encoder = None
        self.weight_raw = None
        self.rel_fc = None
        self.att_fc = None
        self.img_fc = None
        self.char_fc = None
        self.shared_fc = None

        self.gcn_pro = None
        self.rel_pro = None
        self.attr_pro = None
        self.img_pro = None
        self.input_dim = None
        self.entity_emb = None
        self.input_idx = None
        self.n_units = None
        self.n_heads = None
        self.cross_graph_model = None
        self.params = None
        self.optimizer = None
        self.scheduler = None

        self.criterion_cl = None
        self.criterion_align = None

        self.multi_loss_layer = None
        self.align_multi_loss_layer = None
        self.fusion = None  # fusion module

        self.args = parse_args() if args is None else args

        self.set_seed(self.args.seed, self.args.cuda)
        self.device = torch.device(self.args.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, but this torch build has no CUDA. "
                "No CPU fallback was applied."
            )
        self.args.output_root.mkdir(parents=True, exist_ok=True)
        self.init_data()
        self.init_model()
        self.best_hit_1 = -1.0
        self.best_epoch = None
        self.best_data_list = {}
        self.start_epoch = 0
        self.stop_requested = False
        self.stop_signal = None
        self.il_state = ordered_il_state(
            self.train_ill.tolist(), self.test_ill.tolist(),
            self.left_ents, self.right_ents,
        )
        self.left_non_train = list(self.il_state.left_pool)
        self.right_non_train = list(self.il_state.right_pool)
        self.initialize_artifacts()
        if self.args.resume is not None:
            try:
                self.restore_checkpoint(self.args.resume)
            except Exception as exc:
                write_json(self.run_dir/'status.json', {
                    'status': 'failed', 'run_id': self.run_id,
                    'resume_checkpoint': str(self.args.resume),
                    'error': repr(exc),
                })
                raise
        self.print_summary()

    @staticmethod
    def set_seed(seed, cuda=True):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def request_stop(self, signum=None, _frame=None):
        self.stop_requested = True
        self.stop_signal = signum

    def print_summary(self):
        print("-----dataset summary-----")
        print("dataset:\t", self.args.file_dir)
        print("triple num:\t", len(self.triples))
        print("entity num:\t", self.ENT_NUM)
        print("relation num:\t", self.REL_NUM)
        print(
            "train ill num:\t",
            self.train_ill.shape[0],
            "\ttest ill num:\t",
            self.test_ill.shape[0],
        )
        print("-------------------------")

    def forward_encoder(self):
        """Run the encoder, optionally discarding activations for exact recompute."""
        inputs = (
            self.input_idx,
            self.img_features,
            self.rel_features,
            self.att_features,
        )

        def encode(input_idx, image, relation, attribute):
            return self.multimodal_encoder(
                input_idx, image, relation, attribute,
                self.name_features, self.char_features,
            )

        if (self.args.gradient_checkpointing and self.multimodal_encoder.training
                and torch.is_grad_enabled()):
            return activation_checkpoint(
                encode, *inputs, use_reentrant=False, preserve_rng_state=True
            )
        return encode(*inputs)

    def init_data(self):
        # Load data
        lang_list = [1, 2]
        file_dir = self.args.file_dir
        device = self.device
        if (
            "Dualmodal" in self.args.structure_encoder
            or "AblationGNN" in self.args.structure_encoder
        ):
            graph = load_data_dual(file_dir, mcd_self_loops=self.args.mcd_self_loops)

            self.node_size = graph.node_size
            self.rel_size = graph.relation_size
            self.triple_size = graph.triple_size

            print("node_size:\t", self.node_size)
            print("rel_size:\t", self.rel_size)
            print("triple_size:\t", self.triple_size)

            self.adj_matrix = graph.edge_index
            self.rel_matrix = torch.empty((2, 0), dtype=torch.long)
            self.ent_matrix = graph.entity_average
            self.r_index = graph.relation_index
            self.r_val = graph.relation_value
            self.mcd_operator = graph.mcd_operator
            self.graph_diagnostics = graph.diagnostics

        data = load_dataset(self.args.data_root, self.args.dataset,
                            seed=self.args.seed, rate=self.args.rate, rng=np.random)
        self.ent2id_dict, self.triples = data.ent2id, data.triples
        self.r_hs, self.r_ts, self.ids = data.r_hs, data.r_ts, data.ids
        self.left_ents, self.right_ents = data.ordered_ids
        self.ills = [tuple(pair) for pair in np.concatenate([data.split['train_pairs'], data.split['test_pairs']]).tolist()]
        self.data_manifest = data.manifest
        self.data_statistics = data.statistics
        self.image_masks = data.image_masks
        self.split_metadata = data.split_metadata
        self.split_path = save_split(self.args.output_root, self.args.dataset,
                                     self.args.seed, self.args.rate, data.split, data.split_metadata)

        self.ENT_NUM = len(self.ent2id_dict)
        self.REL_NUM = len(self.r_hs)
        print("total ent num: {}, rel num: {}".format(self.ENT_NUM, self.REL_NUM))

        self.img_features = data.image_features

        self.img_features = F.normalize(torch.Tensor(self.img_features).to(device))
        print("image feature shape:", self.img_features.shape)

        data_dir, dataname = os.path.split(file_dir)
        if self.args.word_embedding == "glove":
            word2vec_path = "data/embedding/glove.6B.300d.txt"
        elif self.args.word_embedding == "fasttext":
            pass
        else:
            raise Exception("error word embedding")

        if self.args.dataset in ('zh_en', 'ja_en', 'fr_en'):
            if self.args.w_name or self.args.w_char:
                name_path = os.path.join(
                    data_dir, "translated_ent_name", "dbp_" + dataname + ".json"
                )
                self.ent_vec, self.char_features = load_word_char_features(
                    self.ENT_NUM, word2vec_path, name_path
                )
                self.name_features = F.normalize(torch.Tensor(self.ent_vec)).to(
                    self.device
                )
                self.char_features = F.normalize(
                    torch.Tensor(self.char_features).to(device)
                )
                print("name feature shape:", self.name_features.shape)
                print("char feature shape:", self.char_features.shape)

        # if supervised
        self.train_ill = np.array(
            self.ills[: int(len(self.ills) // 1 * self.args.rate)], dtype=np.int32
        )

        self.test_ill_ = self.ills[int(len(self.ills) // 1 * self.args.rate) :]
        self.test_ill = np.array(self.test_ill_, dtype=np.int32)

        self.test_left = torch.LongTensor(self.test_ill[:, 0]).to(device)
        self.test_right = torch.LongTensor(self.test_ill[:, 1]).to(device)

        left_train = set(self.train_ill[:, 0].tolist())
        right_train = set(self.train_ill[:, 1].tolist())
        self.left_non_train = [value for value in self.left_ents if value not in left_train]
        self.right_non_train = [value for value in self.right_ents if value not in right_train]

        print(
            "#left entity : %d, #right entity: %d"
            % (len(self.left_ents), len(self.right_ents))
        )
        print(
            "#left entity not in train set: %d, #right entity not in train set: %d"
            % (len(self.left_non_train), len(self.right_non_train))
        )
        self.rel_features = data.relation_features
        self.rel_features = torch.Tensor(self.rel_features).to(device)
        print("relation feature shape:", self.rel_features.shape)

        self.att_features = data.attribute_features
        self.att_features = torch.Tensor(self.att_features).to(device)
        print("attribute feature shape:", self.att_features.shape)
        del data

    def init_model(self):
        img_dim = self.img_features.shape[1]
        char_dim = (
            self.char_features.shape[1] if self.char_features is not None else 100
        )
        if "Dualmodal" in self.args.structure_encoder:
            if self.args.structure_encoder == "Dualmodal-joint-LMF":
                self.multimodal_encoder = MultiModalEncoderMrFusion(
                    args=self.args,
                    ent_num=self.node_size,
                    rel_size=self.rel_size,
                    triple_size=self.triple_size,
                    img_feature_dim=img_dim,
                    adj_matrix=self.adj_matrix.to(self.device),
                    r_index=self.r_index.to(self.device),
                    r_val=self.r_val.to(self.device),
                    rel_matrix=self.rel_matrix.to(self.device),
                    ent_matrix=self.ent_matrix.to(self.device),
                    mcd_operator=self.mcd_operator.to(self.device),
                    char_feature_dim=char_dim,
                    use_project_head=self.args.use_project_head,
                    left_ents=self.left_ents,
                    right_ents=self.right_ents,
                ).to(self.device)

        if self.args.use_ms_loss:
            self.multi_loss_layer = CustomMultiLossLayer(
                loss_num=self.args.inner_view_num
            ).to(self.device)
            self.params = [
                {
                    "params": list(self.multimodal_encoder.parameters())
                    + list(self.multi_loss_layer.parameters())
                }
            ]
        else:
            self.params = [{"params": list(self.multimodal_encoder.parameters())}]
        self.optimizer = optim.AdamW(
            self.params, lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        total_params = sum(
            p.numel() for p in self.multimodal_encoder.parameters() if p.requires_grad
        )
        if self.args.use_ms_loss:
            total_params += sum(
                p.numel() for p in self.multi_loss_layer.parameters() if p.requires_grad
            )

        print("total params num", total_params)
        print("MyGram model details:")
        print(self.multimodal_encoder)
        print("optimiser details:")
        print(self.optimizer)

        if self.args.use_sheduler:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=self.args.sheduler_gamma
            )
        elif self.args.use_sheduler_cos:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.args.num_training_steps,
            )

        ms_alpha = self.args.ms_alpha
        ms_beta = self.args.ms_beta
        ms_base = self.args.ms_base

        if self.args.joint_use_nce:
            self.criterion_align = InfoNCE_loss(
                device=self.device, temperature=self.args.tau
            )

        self.criterion_cl = InfoNCE_loss(device=self.device, temperature=self.args.tau)
        self.gram_loss = GRAM_Loss(
            temperature=self.args.gram_temperature,
            K=self.args.gram_k,
            direction_scale=self.args.gram_direction_scale,
            gram_modalities=self.args.gram_modalities,
        )

    def initialize_artifacts(self):
        self.resolved_config = resolved_dict(self.args)
        environment_path = Path(__file__).resolve().parents[1] / 'environment.json'
        environment = (
            json.loads(environment_path.read_text(encoding='utf-8'))
            if environment_path.exists()
            else {'status': 'unavailable'}
        )
        root = Path(__file__).resolve().parents[1]
        source_names = [
            path.relative_to(root).as_posix()
            for path in sorted((root/'src').glob('*.py'))
        ]
        code_revision = {
            'upstream_commit': '8d6bd6f4ddcd807034e53d2b7fccede86b60343c',
            'source_sha256': {name: sha256(root/name) for name in source_names},
        }
        bundle_manifest = root/'source_bundle_manifest.json'
        if bundle_manifest.exists():
            code_revision['source_bundle_manifest_sha256'] = sha256(bundle_manifest)
        runtime_identity = {
            'platform': environment.get('platform') or environment.get('os'),
            'python': environment.get('python'),
            'packages': environment.get('packages'),
            'torch': environment.get('torch'),
        }
        runtime_environment_sha256 = __import__('hashlib').sha256(
            json.dumps(runtime_identity, sort_keys=True, default=str).encode()
        ).hexdigest()
        self.artifact_hashes = {
            'config_sha256': self.resolved_config['config_sha256'],
            'split_sha256': self.split_metadata['split_sha256'],
            'split_npz_sha256': sha256(self.split_path),
            'archive_sha256': self.data_manifest['archive']['sha256'],
            'source_sha256': code_revision['source_sha256'],
            'runtime_environment_sha256': runtime_environment_sha256,
        }
        self.run_id, self.run_dir = initialize_run_directory(
            self.args.output_root,
            self.resolved_config,
            environment,
            self.data_manifest,
            code_revision,
            self.split_path,
        )
        write_json(self.run_dir/'status.json', {
            'status': 'running', 'run_id': self.run_id,
            'resume_checkpoint': str(self.args.resume) if self.args.resume else None,
        })
        append_jsonl(self.run_dir/'environment_segments.jsonl', {
            'event': 'runtime_initialized',
            'runtime_environment_sha256': runtime_environment_sha256,
            'runtime_sha256': self.resolved_config.get('runtime_sha256'),
            'allow_hardware_change': self.args.allow_hardware_change,
            'environment': runtime_identity,
        })

    def training_state(self, epoch_next):
        state = {
            'epoch_next': epoch_next,
            'best_epoch': self.best_epoch,
            'best_hit_1': self.best_hit_1,
            'best_metrics': self.best_data_list,
            'il_state': self.il_state.to_dict(),
        }
        if self.multi_loss_layer is not None:
            state['multi_loss_layer_state_dict'] = self.multi_loss_layer.state_dict()
        return state

    def save_training_checkpoint(self, kind, epoch_next):
        record = save_checkpoint(
            self.run_dir/'checkpoints'/f'{kind}.pt',
            self.multimodal_encoder,
            self.optimizer,
            self.scheduler,
            self.training_state(epoch_next),
            self.resolved_config,
            self.artifact_hashes,
        )
        append_jsonl(self.run_dir/'events.jsonl', {
            'event': 'checkpoint_saved', 'kind': kind,
            'epoch_next': epoch_next, **record,
        })
        return record

    def restore_checkpoint(self, path):
        state = load_checkpoint(
            path,
            self.multimodal_encoder,
            self.optimizer,
            self.scheduler,
            self.resolved_config,
            self.artifact_hashes,
            map_location=self.device,
            allow_hardware_change=self.args.allow_hardware_change,
        )
        self.start_epoch = int(state['epoch_next'])
        self.best_epoch = state['best_epoch']
        self.best_hit_1 = float(state['best_hit_1'])
        self.best_data_list = state['best_metrics']
        if self.multi_loss_layer is not None:
            multi_loss_state = state.get('multi_loss_layer_state_dict')
            if multi_loss_state is None:
                raise ValueError('Checkpoint is missing the multi-loss layer state')
            self.multi_loss_layer.load_state_dict(multi_loss_state)
        self.il_state = IterativeLearningState.from_dict(state['il_state'])
        if self.il_state.fixed_test_links != [tuple(map(int, row)) for row in self.test_ill.tolist()]:
            raise ValueError('Checkpoint fixed test links do not match the persisted split')
        self.train_ill = np.asarray(self.il_state.train_links, dtype=np.int32)
        self.left_non_train = list(self.il_state.left_pool)
        self.right_non_train = list(self.il_state.right_pool)
        append_jsonl(self.run_dir/'events.jsonl', {
            'event': 'checkpoint_loaded', 'path': str(Path(path).resolve()),
            'epoch_next': self.start_epoch,
            'hardware_change_allowed': self.args.allow_hardware_change,
        })

    def semi_supervised_learning(self):
        if not self.il_state.left_pool or not self.il_state.right_pool:
            pairs, diagnostics = mutual_nearest_pairs(
                torch.empty(0, device=self.device),
                self.il_state.left_pool,
                self.il_state.right_pool,
                block_size=self.args.il_block_size,
            )
            diagnostics['model_mode'] = self.args.il_model_mode
            return pairs, diagnostics
        was_training = self.multimodal_encoder.training
        if self.args.il_model_mode == 'eval':
            self.multimodal_encoder.eval()
        with torch.no_grad():
            gph_emb1,gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = self.forward_encoder()

            pairs, diagnostics = mutual_nearest_pairs(
                joint_emb,
                self.il_state.left_pool,
                self.il_state.right_pool,
                block_size=self.args.il_block_size,
            )
            del gph_emb1, gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb
        if self.args.il_model_mode == 'eval' and was_training:
            self.multimodal_encoder.train()
        diagnostics['model_mode'] = self.args.il_model_mode
        return pairs, diagnostics

    def alignment_loss(
        self,
        joint_emb,
        gph_emb,
        gph_emb1,
        rel_emb,
        att_emb,
        img_emb,
        name_emb,
        char_emb,
        train_ill,
    ):
        loss, diagnostics = alignment_objective(
            joint_emb=joint_emb,
            structure_emb=gph_emb1,
            img_emb=img_emb,
            attr_emb=att_emb,
            rel_emb=rel_emb,
            train_links=train_ill,
            nce_loss=self.criterion_align,
            gram_loss=self.gram_loss,
            gram_weight=self.args.gram_weight,
            disable_gram=self.args.disable_gram,
        )
        self.last_loss_diagnostics = diagnostics
        print(' loss_diagnostics=' + json.dumps(diagnostics, sort_keys=True), end='')
        return loss

    def train(self):

        # print args
        print("model config is:")
        pprint(self.args, indent=2)

        # Train
        print("[start training...] ")
        t_total = time.time()
        epochs_this_process = 0

        bsize = self.args.bsize
        device = self.device

        self.input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(device)

        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_started = time.time()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            self.optimizer, optimizer_event = apply_optimizer_policy(
                self.optimizer, self.params, self.args, epoch
            )
            if optimizer_event is not None:
                optimizer_record = {
                    'epoch': epoch,
                    'event': optimizer_event,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'optimizer_state_items': len(self.optimizer.state),
                }
                append_jsonl(self.run_dir/'events.jsonl', optimizer_record)
                print(' optimizer_event=' + json.dumps(optimizer_record, sort_keys=True))

            self.multimodal_encoder.train()
            if self.args.use_ms_loss:
                self.multi_loss_layer.train()
            self.optimizer.zero_grad()

            gph_emb1,gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = self.forward_encoder()

            def shape_of(value):
                return None if value is None else tuple(value.shape)

            print("gph_emb", shape_of(gph_emb))
            print("img_emb", shape_of(img_emb))
            print("rel_emb", shape_of(rel_emb))
            print("att_emb", shape_of(att_emb))
            print("joint_emb", shape_of(joint_emb))


            loss_sum_all = 0.0
            epoch_loss_diagnostics = []

            np.random.shuffle(self.train_ill)
            self.il_state.train_links = [tuple(map(int, row)) for row in self.train_ill.tolist()]
            print("train_ill length:", len(self.train_ill))
            for si in np.arange(0, self.train_ill.shape[0], bsize):
                if self.args.use_joint_loss:
                    loss_all = self.alignment_loss(
                        joint_emb,
                        gph_emb,
                        gph_emb1,
                        rel_emb,
                        att_emb,
                        img_emb,
                        name_emb,
                        char_emb,
                        self.train_ill[si : si + bsize],
                    )
                else:
                    continue
                if loss_all is None:
                    continue
                print(" loss_all: {:f},".format(loss_all))
                retain_graph = si + bsize < self.train_ill.shape[0]
                loss_all.backward(retain_graph=retain_graph)
                loss_sum_all += float(loss_all.detach())
                epoch_loss_diagnostics.append(dict(self.last_loss_diagnostics))

            grad_squared = sum(
                float(parameter.grad.detach().float().square().sum())
                for parameter in self.multimodal_encoder.parameters()
                if parameter.grad is not None
            )
            grad_norm = grad_squared ** 0.5
            for diagnostics in epoch_loss_diagnostics:
                diagnostics['grad_norm'] = grad_norm
            print(' epoch_loss_diagnostics=' + json.dumps(epoch_loss_diagnostics, sort_keys=True))

            self.optimizer.step()
            print("[epoch {:d}] loss_all: {:f}".format(epoch, loss_sum_all))

            if self.args.use_sheduler and epoch > 400:
                print("use_sheduler", self.optimizer, self.scheduler)
                self.scheduler.step()

            if self.args.use_sheduler_cos:
                print("use_sheduler_cos", self.optimizer, self.scheduler)
                self.scheduler.step()
            if (self.args.il and epoch >= self.args.il_start
                    and (epoch + 1) % self.args.il_step == 0):
                mutual_pairs, mining = self.semi_supervised_learning()
                label_diagnostic = diagnostic_precision(
                    mutual_pairs, self.il_state.fixed_test_links
                )
                il_event = update_il_state(
                    self.il_state, mutual_pairs, epoch,
                    self.args.il_start, self.args.il_step,
                )
                self.train_ill = np.asarray(self.il_state.train_links, dtype=np.int32)
                self.left_non_train = list(self.il_state.left_pool)
                self.right_non_train = list(self.il_state.right_pool)
                append_jsonl(self.run_dir/'events.jsonl', {
                    'event': 'iterative_learning', **il_event,
                    'mining': mining, 'held_out_diagnostic': label_diagnostic,
                })
                print(' il_event=' + json.dumps(il_event, sort_keys=True))
            if ((epoch + 1) % self.args.check_point == 0
                    or epoch + 1 == self.args.epochs):
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                self.test(epoch)

            append_jsonl(self.run_dir/'metrics.jsonl', {
                'epoch': epoch,
                'phase': 'train',
                'loss_sum_of_batch_means': loss_sum_all,
                'loss_batches': epoch_loss_diagnostics,
                'grad_norm': grad_norm,
                'lr': self.optimizer.param_groups[0]['lr'],
                'train_pairs': len(self.il_state.train_links),
                'runtime_seconds': time.time() - epoch_started,
                'cuda_max_memory_allocated': (
                    torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == 'cuda' else None
                ),
                'cuda_max_memory_reserved': (
                    torch.cuda.max_memory_reserved(self.device)
                    if self.device.type == 'cuda' else None
                ),
            })
            self.save_training_checkpoint('last', epoch + 1)

            del joint_emb, gph_emb1, gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb

            epochs_this_process += 1
            wall_limit_reached = (
                self.args.max_wall_minutes > 0
                and time.time() - t_total >= self.args.max_wall_minutes * 60
            )
            epoch_limit_reached = (
                self.args.max_epochs_this_process > 0
                and epochs_this_process >= self.args.max_epochs_this_process
            )
            if self.stop_requested or wall_limit_reached or epoch_limit_reached:
                reason = (
                    f'signal_{self.stop_signal}' if self.stop_requested
                    else 'max_wall_minutes' if wall_limit_reached
                    else 'max_epochs_this_process'
                )
                write_json(self.run_dir/'status.json', {
                    'status': 'paused', 'run_id': self.run_id,
                    'epoch_next': epoch + 1, 'reason': reason,
                })
                append_jsonl(self.run_dir/'events.jsonl', {
                    'event': 'run_paused', 'epoch': epoch,
                    'epoch_next': epoch + 1, 'reason': reason,
                })
                print(f'[run paused safely] epoch_next={epoch + 1} reason={reason}', flush=True)
                return 'paused'

        print("[optimization finished!]")
        print(
            "best epoch is {}, hits@1 hits@10 MRR MR is: {}\n".format(
                self.best_epoch, self.best_data_list
            )
        )
        print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))
        write_json(self.run_dir/'summary.json', {
            'run_id': self.run_id,
            'completed_epochs': self.args.epochs,
            'best_epoch': self.best_epoch,
            'best_test': self.best_data_list,
            'candidate_protocol': 'test_pairs',
            'candidate_semantics': 'ordered positional rows; duplicate entity IDs retained',
            'selection_rule': 'best_test_hits@1_l2r',
            'runtime_seconds': time.time() - t_total,
        })
        write_json(self.run_dir/'status.json', {
            'status': 'completed', 'run_id': self.run_id,
            'completed_epochs': self.args.epochs,
        })
        write_json(self.run_dir/'COMPLETED.json', {
            'schema_version': 1, 'status': 'completed', 'run_id': self.run_id,
            'epoch_next': self.args.epochs,
            'experiment_sha256': self.resolved_config['experiment_sha256'],
            'artifact_hashes': self.artifact_hashes,
        })
        return 'completed'

    def test(self, epoch):
        start = time.time()
        was_training = self.multimodal_encoder.training
        self.multimodal_encoder.eval()
        with torch.no_grad():
            outputs = self.forward_encoder()
            result = evaluate_embeddings(
                outputs[-1],
                self.test_left,
                self.test_right,
                csls_k=self.args.csls_k if self.args.csls else None,
                block_size=self.args.eval_block_size,
                top_k=(1, 10, 50),
            )
        if was_training:
            self.multimodal_encoder.train()
        record = {
            'epoch': epoch,
            'phase': 'evaluation',
            'metrics': result['metrics'],
            'protocol': result['protocol'],
            'runtime_seconds': time.time() - start,
            'best_test_selection_bias': True,
        }
        append_jsonl(self.run_dir/'metrics.jsonl', record)
        write_predictions(
            self.run_dir/'predictions'/'last_l2r.tsv', result['predictions']['l2r']
        )
        write_predictions(
            self.run_dir/'predictions'/'last_r2l.tsv', result['predictions']['r2l']
        )
        l2r = result['metrics']['l2r']
        if l2r['hits@1'] > self.best_hit_1:
            self.best_hit_1 = l2r['hits@1']
            self.best_epoch = epoch
            self.best_data_list = result['metrics']
            write_predictions(
                self.run_dir/'predictions'/'best_test_l2r.tsv',
                result['predictions']['l2r'],
            )
            write_predictions(
                self.run_dir/'predictions'/'best_test_r2l.tsv',
                result['predictions']['r2l'],
            )
            self.save_training_checkpoint('best_test', epoch + 1)
        print(' evaluation=' + json.dumps(record, sort_keys=True))
        return result


def main(argv=None):
    try:
        import sys
        sys.stdout.reconfigure(errors='backslashreplace')
    except AttributeError:
        pass
    args = parse_args(argv)
    resolved = emit_resolved(args)
    if args.print_config:
        print(__import__('json').dumps(resolved, indent=2, ensure_ascii=False))
        return 0
    model = None
    try:
        model = MyGram(args)
        signal.signal(signal.SIGINT, model.request_stop)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, model.request_stop)
        outcome = model.train()
    except Exception as exc:
        if model is not None and hasattr(model, 'run_dir'):
            write_json(model.run_dir/'status.json', {
                'status': 'failed', 'run_id': model.run_id,
                'error': repr(exc),
            })
        raise
    return 75 if outcome == 'paused' else 0


if __name__ == "__main__":
    raise SystemExit(main())
