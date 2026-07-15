"""
src/utils/data_loader.py

Dataset discovery, stratified splitting, and tf.data pipeline for IrisNet.

Usage
-----
    from src.utils.data_loader import build_datasets, NUM_CLASSES

    train_ds, val_ds, test_ds = build_datasets(
        processed_root='data/processed',
        batch_size=32,
        arcface=False,   # True → yields (image, label_onehot) for ArcFace
    )

Design decisions
----------------
* Class label = one unique (subset, subject, eye) folder path  →  up to 4 115
  classes from the 30 626 preprocessed tensors.
* Identities with only 1 file are added to train only (cannot validate/test).
* Identities with exactly 2 files get train + test (skip val); val gets
  duplicated from train to avoid an empty dataset for that identity.
* Identities with >= 3 files receive a proper stratified 70/20/10 split.
* Files are loaded lazily via tf.data.Dataset.map so nothing is pre-loaded
  into RAM.
* The exact test split is serialised to data/test_split.json for Phase 6.

Split modes
-----------
* 'stratified'        (default) — closed-set: per-identity 70/20/10 sample split.
                      Identities appear in all splits; test samples come from
                      identities the model has seen during training.
* 'identity_disjoint' — open-set: partitions identities (not samples). A
                      disjoint subset of identities is held out entirely for
                      test; the model never sees these identities during
                      training. This tests generalisation to unseen subjects.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.20
# TEST_FRAC  = 0.10  (remainder)

SEED = 42
IMG_SHAPE = (128, 128, 1)

TEST_SPLIT_PATH = 'data/test_split.json'
OPENSET_SPLIT_PATH = 'data/test_split_openset.json'

# Identity-disjoint split: fraction of identities reserved for test.
# The remaining identities are split stratified-by-sample into train/val.
OPENSET_TEST_IDENT_FRAC = 0.10
OPENSET_VAL_FRAC        = 0.20   # sample-level, on non-test identities


# ── 1. Discover all .npy files and assign integer class labels ────────────────

def _normalise_processed_roots(processed_root: Union[str, list, tuple]) -> List[str]:
    """Return a stable list of processed roots from a string or sequence."""
    if isinstance(processed_root, (list, tuple)):
        roots = [str(r) for r in processed_root]
    else:
        roots = [str(processed_root)]
    return [r for r in roots if r]


def _discover(processed_root: Union[str, list, tuple]):
    """Walk processed_root and return (paths, int_labels, label_to_idx).

    A 'class' is the subdirectory directly under processed_root that
    contains .npy files (e.g. CASIA-Iris-Interval/001/L).
    """
    roots = [Path(r) for r in _normalise_processed_roots(processed_root)]
    # Map identity folder → sorted list of .npy paths
    identity_files: dict = {}
    seen_rel_files = set()
    for root in roots:
        if not root.exists():
            print(f'[data_loader] WARN missing processed root: {root}')
            continue
        for path in sorted(root.rglob('*.npy')):
            identity = path.parent.relative_to(root).as_posix()
            rel_file = Path(identity) / path.name
            rel_key = rel_file.as_posix()
            if rel_key in seen_rel_files:
                # Prefer earlier roots, normally data/processed over recovered.
                continue
            seen_rel_files.add(rel_key)
            identity_files.setdefault(identity, []).append(str(path))

    # Stable, sorted label assignment
    sorted_identities = sorted(identity_files.keys())
    label_to_idx = {ident: idx for idx, ident in enumerate(sorted_identities)}

    all_paths: List[str] = []
    all_labels: List[int] = []
    for ident in sorted_identities:
        files = sorted(identity_files[ident])
        lbl = label_to_idx[ident]
        all_paths.extend(files)
        all_labels.extend([lbl] * len(files))

    return all_paths, all_labels, label_to_idx, identity_files


def _stratified_split(identity_files: dict, label_to_idx: dict, rng: random.Random):
    """Return train/val/test lists of (path, label) tuples with stratified split.

    Rules:
      >=  3 samples: 70/20/10 per-identity (at least 1 per split)
      == 2 samples: 1 train, 1 test  (val borrows the train sample)
      == 1 sample:  train only
    """
    train, val, test = [], [], []
    for ident, files in identity_files.items():
        files = sorted(files)
        lbl = label_to_idx[ident]
        n = len(files)
        shuffled = files[:]
        rng.shuffle(shuffled)

        if n == 1:
            train.append((shuffled[0], lbl))
        elif n == 2:
            train.append((shuffled[0], lbl))
            test.append((shuffled[1], lbl))
        else:
            n_test  = max(1, round(n * (1 - TRAIN_FRAC - VAL_FRAC)))
            n_val   = max(1, round(n * VAL_FRAC))
            n_train = n - n_val - n_test

            train_files = shuffled[:n_train]
            val_files   = shuffled[n_train:n_train + n_val]
            test_files  = shuffled[n_train + n_val:]

            train.extend([(f, lbl) for f in train_files])
            val.extend(  [(f, lbl) for f in val_files])
            test.extend( [(f, lbl) for f in test_files])

    return train, val, test


def _identity_disjoint_split(
    identity_files: dict,
    rng: random.Random,
    test_ident_frac: float = OPENSET_TEST_IDENT_FRAC,
    val_sample_frac: float = OPENSET_VAL_FRAC,
    test_idents: set = None,
):
    """Partition identities (not samples) for open-set evaluation.

    Rules:
      * Test identities are picked only from identities with >= 2 samples
        (need at least 2 per identity to form genuine pairs).
      * All samples of a test identity go to test (model never sees them).
      * Remaining identities are sample-split: (1 - val_sample_frac) train,
        val_sample_frac val. Singleton identities go to train only.

    Returns:
        train, val, test         — lists of (path, label) tuples
        train_label_to_idx       — maps identity-string → class index (0..N-1)
                                   — only non-test identities get indices.
        test_local_label_to_idx  — maps identity-string → local test index
                                   — for grouping test samples by identity
                                   during pair generation (not used by model).
    """
    if test_idents is None:
        test_idents = select_identity_disjoint_test_identities(
            identity_files, rng, test_ident_frac,
        )
    else:
        test_idents = set(test_idents)
    train_pool_idents = sorted(set(identity_files.keys()) - test_idents)

    # Relabel train pool: 0..N-1
    train_label_to_idx = {ident: idx for idx, ident in enumerate(train_pool_idents)}

    # Relabel test pool with a separate local index 0..M-1 (for pair grouping)
    test_idents_sorted = sorted(test_idents)
    test_local_label_to_idx = {ident: idx for idx, ident in enumerate(test_idents_sorted)}

    train, val, test = [], [], []

    # Train/val identities: sample-stratified split
    for ident in train_pool_idents:
        files = sorted(identity_files[ident])
        lbl = train_label_to_idx[ident]
        n = len(files)
        shuffled = files[:]
        rng.shuffle(shuffled)

        if n == 1:
            train.append((shuffled[0], lbl))
        else:
            n_val   = max(1, round(n * val_sample_frac))
            n_train = n - n_val
            train.extend((f, lbl) for f in shuffled[:n_train])
            val.extend(  (f, lbl) for f in shuffled[n_train:])

    # Test identities: all samples go to test, with local labels
    for ident in test_idents_sorted:
        files = sorted(identity_files[ident])
        lbl = test_local_label_to_idx[ident]
        test.extend((f, lbl) for f in files)

    return train, val, test, train_label_to_idx, test_local_label_to_idx


def select_identity_disjoint_test_identities(
    identity_files: dict,
    rng: random.Random,
    test_ident_frac: float = OPENSET_TEST_IDENT_FRAC,
) -> set:
    """Choose identity-disjoint test identities from identities with >=2 files."""
    eligible = [ident for ident, files in identity_files.items() if len(files) >= 2]
    eligible_sorted = sorted(eligible)
    rng.shuffle(eligible_sorted)
    n_test_id = max(1, round(len(eligible_sorted) * test_ident_frac))
    return set(eligible_sorted[:n_test_id])


# ── 2. tf.data.Dataset factory ────────────────────────────────────────────────

def _make_tf_dataset(
    samples: List[Tuple[str, int]],
    num_classes: int,
    batch_size: int,
    augment: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    """Build a batched tf.data.Dataset from a list of (path, label) pairs.

    Labels are one-hot float32 in all cases — both the softmax head and the
    ArcFace head consume CategoricalCrossentropy, so the format is identical.

    Args:
        samples:     list of (npy_path_str, int_label)
        num_classes: total number of identity classes
        batch_size:  samples per batch
        augment:     apply RandomRotation augmentation
        shuffle:     shuffle before each epoch
    """
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    def load_npy(path):
        """Load one (128,128,1) float32 tensor from an .npy file."""
        img = tf.numpy_function(
            func=lambda p: np.load(p.decode()).astype(np.float32),
            inp=[path],
            Tout=tf.float32,
        )
        img.set_shape(IMG_SHAPE)
        return img

    img_ds = path_ds.map(load_npy, num_parallel_calls=tf.data.AUTOTUNE)

    # One-hot labels (both softmax and arcface use categorical cross-entropy)
    def to_onehot(lbl):
        return tf.one_hot(lbl, depth=num_classes, dtype=tf.float32)

    label_oh_ds = label_ds.map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)

    ds = tf.data.Dataset.zip((img_ds, label_oh_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(samples), 4096), seed=SEED,
                        reshuffle_each_iteration=True)

    if augment:
        augmentor = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(factor=0.05, fill_mode='reflect'),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.05, 0.05),
                width_factor=(-0.05, 0.05),
                fill_mode='reflect',
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.03,
                width_factor=0.03,
                fill_mode='reflect',
            ),
            tf.keras.layers.GaussianNoise(0.01),
        ])
        ds = ds.map(
            lambda x, y: (augmentor(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── 3. Public API ─────────────────────────────────────────────────────────────

# Global — set after first build_datasets() call
NUM_CLASSES: int = 0


def build_datasets(
    processed_root: Union[str, list, tuple] = 'data/processed',
    batch_size: int = 32,
    test_split_path: str = None,
    min_samples: int = 1,
    split_mode: str = 'stratified',
    fixed_test_identities: list = None,
):
    """Discover data, split, and return three tf.data.Dataset objects.

    Labels are always one-hot float32; both the softmax head and the ArcFace
    head consume CategoricalCrossentropy so no format difference is needed.

    Args:
        processed_root:  path to data/processed/ or a list of roots. When a list
                         is passed, roots are merged by identity relative path;
                         earlier roots win duplicate files.
        batch_size:      batch size for all three splits
        test_split_path: where to write the test split JSON (defaults to
                         TEST_SPLIT_PATH for 'stratified' and
                         OPENSET_SPLIT_PATH for 'identity_disjoint')
        min_samples:     minimum images per identity to include (default 1 =
                         keep all; set to 2 for ArcFace to exclude singletons)
        split_mode:      'stratified' — closed-set, identities in all splits
                         'identity_disjoint' — open-set, disjoint test identities
        fixed_test_identities:
                         optional identity strings to hold out for
                         identity_disjoint mode. This keeps Softmax and ArcFace
                         on the same open-set test identities even when ArcFace
                         filters singleton training classes.

    Returns:
        (train_ds, val_ds, test_ds, num_classes)
        For identity_disjoint: num_classes = train identities only.
    """
    global NUM_CLASSES

    if test_split_path is None:
        test_split_path = (OPENSET_SPLIT_PATH if split_mode == 'identity_disjoint'
                           else TEST_SPLIT_PATH)

    roots = _normalise_processed_roots(processed_root)
    _, _, label_to_idx, identity_files_all = _discover(roots)

    if not identity_files_all:
        raise RuntimeError(f'No .npy files found under processed roots: {roots}')

    identity_files = identity_files_all

    # Filter out identities with fewer than min_samples images
    if min_samples > 1:
        identity_files = {k: v for k, v in identity_files.items()
                          if len(v) >= min_samples}
        sorted_identities = sorted(identity_files.keys())
        label_to_idx = {ident: idx for idx, ident in enumerate(sorted_identities)}
        print(f'[data_loader] Filtered to identities with >= {min_samples} samples')

    rng = random.Random(SEED)

    if split_mode == 'identity_disjoint':
        if fixed_test_identities is None:
            split_rng = random.Random(SEED)
            fixed_test_identities = sorted(select_identity_disjoint_test_identities(
                identity_files_all, split_rng,
            ))
        (train_samples, val_samples, test_samples,
         train_label_to_idx, test_local_label_to_idx) = _identity_disjoint_split(
            identity_files, rng, test_idents=set(fixed_test_identities),
        )
        num_classes = len(train_label_to_idx)
        NUM_CLASSES = num_classes
        n_test_id = len(test_local_label_to_idx)

        _save_test_split(
            test_samples, test_local_label_to_idx, test_split_path,
            num_classes_model=num_classes,
            processed_roots=roots,
            train_label_to_idx=train_label_to_idx,
        )
        print(f'[data_loader] Mode         : identity_disjoint (open-set)')
        print(f'[data_loader] Train idents : {num_classes}')
        print(f'[data_loader] Test idents  : {n_test_id} (disjoint from train)')
    elif split_mode == 'stratified':
        num_classes = len(label_to_idx)
        NUM_CLASSES = num_classes
        train_samples, val_samples, test_samples = _stratified_split(
            identity_files, label_to_idx, rng
        )
        _save_test_split(
            test_samples, label_to_idx, test_split_path,
            num_classes_model=num_classes,
            processed_roots=roots,
        )
        print(f'[data_loader] Mode         : stratified (closed-set)')
        print(f'[data_loader] Classes      : {num_classes}')
    else:
        raise ValueError(f'Unknown split_mode: {split_mode!r}')

    print(f'[data_loader] Train samples: {len(train_samples)}')
    print(f'[data_loader] Val   samples: {len(val_samples)}')
    print(f'[data_loader] Test  samples: {len(test_samples)}')

    train_ds = _make_tf_dataset(train_samples, num_classes, batch_size,
                                augment=True,  shuffle=True)
    val_ds   = _make_tf_dataset(val_samples,   num_classes, batch_size,
                                augment=False, shuffle=False)
    test_ds  = _make_tf_dataset(test_samples,  num_classes, batch_size,
                                augment=False, shuffle=False)

    return train_ds, val_ds, test_ds, num_classes


def _save_test_split(test_samples, label_to_idx, path, num_classes_model=None,
                     processed_roots=None, train_label_to_idx=None):
    """Serialise the test split to JSON for reproducible Phase 6 evaluation.

    Args:
        test_samples:      list of (path, label) tuples.
        label_to_idx:      identity-string → label-index mapping used in samples.
                           For open-set, this is the local test-identity index.
        path:              output JSON path.
        num_classes_model: number of classes the *trained model* has (used for
                           softmax embedding extraction). Defaults to
                           len(label_to_idx) for backward compatibility with
                           the closed-set split.
    """
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    records = [
        {'path': p, 'label_idx': lbl, 'identity': idx_to_label[lbl]}
        for p, lbl in test_samples
    ]
    if num_classes_model is None:
        num_classes_model = len(label_to_idx)
    payload = {
        'num_classes': num_classes_model,
        'samples': records,
    }
    if processed_roots is not None:
        payload['processed_roots'] = processed_roots
    if train_label_to_idx is not None:
        payload['train_identities'] = sorted(train_label_to_idx.keys())
        payload['test_identities'] = sorted(label_to_idx.keys())
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[data_loader] Test split saved -> {path}  ({len(records)} samples)')


def load_test_split(test_split_path: str = TEST_SPLIT_PATH):
    """Load the persisted test split JSON and return (paths, int_labels, num_classes)."""
    with open(test_split_path) as f:
        data = json.load(f)
    paths  = [s['path']      for s in data['samples']]
    labels = [s['label_idx'] for s in data['samples']]
    return paths, labels, data['num_classes']
