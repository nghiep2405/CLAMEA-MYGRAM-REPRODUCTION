try:
    from gdrive_fsspec import GoogleDriveFileSystem
except ImportError:
    run_checked([
        sys.executable, "-m", "pip", "install",
        "-q", "gdrive-fsspec"
    ])
    from gdrive_fsspec import GoogleDriveFileSystem

fs_drive = GoogleDriveFileSystem(
    root_file_id=PROJECT_FOLDER_ID,
    token="cache",
    auth_kwargs={"use_local_webserver": False},
    use_listings_cache=False,
    skip_instance_cache=True,
)

assert fs_drive.exists(DATA_NAME), (
    f"{DATA_NAME} chưa có trong Drive folder "
    f"{PROJECT_FOLDER_ID}"
)

local_zip = WORK / DATA_NAME

if (
    not local_zip.is_file()
    or file_sha256(local_zip) != DATA_SHA256
):
    print("Downloading 236 MB archive directly from Drive...")
    fs_drive.get_file(DATA_NAME, str(local_zip))

assert local_zip.stat().st_size == 236332586, (
    "Dataset ZIP size mismatch"
)

assert file_sha256(local_zip) == DATA_SHA256, (
    "Dataset ZIP SHA-256 mismatch"
)

# Verify entire ZIP, including an already extracted archive.
with zipfile.ZipFile(local_zip) as verified_archive:
    assert verified_archive.testzip() is None, (
        "ZIP CRC failure"
    )

    for ent in verified_archive.infolist():
        destination = (
            WORK / "datasets" / ent.filename
        ).resolve()

        assert destination.is_relative_to(
            (WORK / "datasets").resolve()
        ), "Unsafe ZIP path"

unpacked_root = WORK / "datasets"
data_dir = unpacked_root / "CLAMEA_FBDB15K_protocolB"

if not (data_dir / "721train_links").is_file():
    unpacked_root.mkdir(exist_ok=True)

    with zipfile.ZipFile(local_zip) as archive:
        assert archive.testzip() is None, (
            "ZIP CRC failure"
        )

        for entry in archive.infolist():
            target = (
                unpacked_root / entry.filename
            ).resolve()

            assert target.is_relative_to(
                unpacked_root.resolve()
            ), "Unsafe ZIP path"

        archive.extractall(unpacked_root)

required = {
    "ent_ids_1",
    "ent_ids_2",
    "rel_triples_1",
    "rel_triples_2",
    "attr_triples_1",
    "attr_triples_2",
    "image.pkl",
    "attr_embeddings_01_768d.pkl",
    "attr_embeddings_02_768d.pkl",
    "721train_links",
    "721valid_links",
    "721test_links",
}

assert required.issubset({
    p.name for p in data_dir.iterdir()
}), "Dataset files missing"

split_sets = []

for split_name, (
    wanted_hash,
    wanted_count,
) in EXPECTED_SPLITS.items():

    p = data_dir / split_name

    assert file_sha256(p) == wanted_hash, (
        "Split hash changed: " + split_name
    )

    pairs = [
        tuple(s.split("\t"))
        for s in p.read_text().splitlines()
        if s.strip()
    ]

    assert len(pairs) == wanted_count, split_name
    assert all(len(x) == 2 for x in pairs), split_name
    assert len(set(pairs)) == len(pairs), (
        split_name + " has duplicates"
    )

    split_sets.append(set(pairs))

for ia in range(3):
    for ib in range(ia + 1, 3):
        assert not split_sets[ia].intersection(
            split_sets[ib]
        ), "Pair leakage between splits"

        for side in (0, 1):
            left = {
                p[side] for p in split_sets[ia]
            }
            right = {
                p[side] for p in split_sets[ib]
            }

            assert not (left & right), (
                "Entity leakage between splits"
            )

assert sum(len(p) for p in split_sets) == 12846

DATA_CHECK = {
    "zip_sha256": DATA_SHA256,
    "dataset_path": str(data_dir),
    "n_train": 2569,
    "n_valid": 1284,
    "n_test": 8993,
    "entity_disjoint": True,
    "data_kind": (
        "HUMEA/MMKB; VGG16 4096d; Protocol-B"
    ),
}

print("DATA PASS:")
print(json.dumps(DATA_CHECK, indent=2))
