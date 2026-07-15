"""
src/models/train_softmax.py

Softmax baseline training script for IrisNet.

This trains IrisNet with a standard Dense + softmax classification head and
categorical cross-entropy loss.  It serves as the closed-set accuracy baseline
before the ArcFace variant is trained.

Saved artefacts
---------------
  models/softmax_best.h5      Best checkpoint (lowest val_loss)
  models/softmax_history.json Training history (loss + accuracy per epoch)

Usage
-----
    python -m src.models.train_softmax
    python -m src.models.train_softmax --epochs 5 --batch_size 16
    python -m src.models.train_softmax --cpu        # force CPU (disables DirectML)
"""

import argparse
import json
import os

import tensorflow as tf

from src.models.architecture import build_irisnet
from src.utils.data_loader import build_datasets

# ── Hyper-parameters (can be overridden via CLI) ──────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 32
LR_INITIAL  = 1e-3
EMBEDDING_DIM = 512

CHECKPOINT_PATH         = 'models/softmax_best.h5'
HISTORY_PATH            = 'models/softmax_history.json'
OPENSET_CHECKPOINT_PATH = 'models/softmax_openset_best.h5'
OPENSET_HISTORY_PATH    = 'models/softmax_openset_history.json'
EXPANDED_OPENSET_CHECKPOINT_PATH = 'models/softmax_expanded_openset_best.h5'
EXPANDED_OPENSET_HISTORY_PATH    = 'models/softmax_expanded_openset_history.json'
EXPANDED_OPENSET_SPLIT_PATH      = 'data/test_split_expanded_openset.json'
EXPANDED_ROOTS = ['data/processed', 'data/processed_recovered']


def build_softmax_model(num_classes: int, embedding_dim: int = EMBEDDING_DIM):
    """Attach a softmax classification head to the IrisNet backbone.

    The backbone output is the L2-normalised 512-D embedding.
    A Dense(num_classes, activation='softmax') head is added on top.

    Args:
        num_classes:   number of identity classes in the training set
        embedding_dim: embedding dimension of the IrisNet backbone

    Returns:
        Compiled tf.keras.Model ready for model.fit()
    """
    backbone = build_irisnet(input_shape=(128, 128, 1), embedding_dim=embedding_dim)

    # Freeze nothing — train end-to-end
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='softmax_head',
    )(backbone.output)

    model = tf.keras.Model(
        inputs=backbone.input,
        outputs=outputs,
        name='IrisNet_softmax',
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INITIAL),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        jit_compile=False,
    )
    return model


def _load_history(history_path: str):
    if not os.path.isfile(history_path):
        return {}
    with open(history_path, 'r') as f:
        return json.load(f)


def _best_history_value(history_data: dict, metric: str, mode: str):
    values = history_data.get(metric, [])
    if not values:
        return None
    return min(values) if mode == 'min' else max(values)


def _merge_history(previous: dict, current: dict, initial_epoch: int):
    merged = {}
    for key in set(previous) | set(current):
        prior_values = list(previous.get(key, []))
        current_values = list(current.get(key, []))
        merged[key] = prior_values[:initial_epoch] + current_values
    return merged


def get_callbacks(checkpoint_path: str = CHECKPOINT_PATH,
                  initial_value_threshold=None):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            initial_value_threshold=initial_value_threshold,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def _select_training_paths(openset: bool, expanded: bool):
    if expanded and not openset:
        raise ValueError('Expanded training is currently defined for --openset only')
    if expanded:
        return (
            EXPANDED_OPENSET_CHECKPOINT_PATH,
            EXPANDED_OPENSET_HISTORY_PATH,
            EXPANDED_OPENSET_SPLIT_PATH,
            EXPANDED_ROOTS,
        )
    return (
        OPENSET_CHECKPOINT_PATH if openset else CHECKPOINT_PATH,
        OPENSET_HISTORY_PATH if openset else HISTORY_PATH,
        None,
        ['data/processed'],
    )


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, cpu: bool = False,
          openset: bool = False, expanded: bool = False,
          processed_roots: list = None, resume: bool = False,
          initial_epoch: int = 0, show_summary: bool = False):
    if cpu:
        tf.config.set_visible_devices([], 'GPU')
        print('[train_softmax] GPU disabled — running on CPU')

    checkpoint_path, history_path, split_path, default_roots = _select_training_paths(
        openset, expanded,
    )
    processed_root = processed_roots if processed_roots else default_roots
    split_mode = 'identity_disjoint' if openset else 'stratified'
    min_samples = 1  # Softmax handles singleton identities fine

    print('=' * 60)
    mode = 'expanded open-set' if expanded else ('open-set' if openset else 'closed-set')
    print(f'IrisNet — Softmax Training  ({mode})')
    print('=' * 60)
    print(f'Processed roots: {processed_root}')

    train_ds, val_ds, _, num_classes = build_datasets(
        processed_root=processed_root,
        batch_size=batch_size,
        split_mode=split_mode,
        min_samples=min_samples,
        test_split_path=split_path,
    )
    print(f'Classes: {num_classes}  |  Batch size: {batch_size}  |  Epochs: {epochs}')

    model = build_softmax_model(num_classes)
    if resume and os.path.isfile(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f'[train_softmax] Resumed weights from {checkpoint_path}')
    if show_summary:
        model.summary()

    previous_history = _load_history(history_path) if resume else {}
    checkpoint_threshold = _best_history_value(previous_history, 'val_loss', 'min')
    if checkpoint_threshold is not None:
        print(f'[train_softmax] Checkpoint resumes from best val_loss={checkpoint_threshold:.6f}')

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=get_callbacks(
            checkpoint_path,
            initial_value_threshold=checkpoint_threshold,
        ),
        verbose=2,
    )

    # Persist history for the notebook
    merged_history = _merge_history(previous_history, history.history, initial_epoch)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(merged_history, f, indent=2)
    print(f'History saved -> {history_path}')
    print(f'Best model saved -> {checkpoint_path}')
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,            default=EPOCHS)
    parser.add_argument('--batch_size', type=int,            default=BATCH_SIZE)
    parser.add_argument('--cpu',        action='store_true', default=False)
    parser.add_argument('--openset',    action='store_true', default=False,
                        help='Train with identity-disjoint open-set split')
    parser.add_argument('--expanded',   action='store_true', default=False,
                        help='Use data/processed + data/processed_recovered for open-set training')
    parser.add_argument('--processed-root', action='append', dest='processed_roots',
                        help='Processed root to use; repeat to merge roots')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Load the selected checkpoint before training')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='Initial epoch passed to model.fit for chunked runs')
    parser.add_argument('--summary', action='store_true', default=False,
                        help='Print model.summary() before training')
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, cpu=args.cpu,
          openset=args.openset, expanded=args.expanded,
          processed_roots=args.processed_roots, resume=args.resume,
          initial_epoch=args.initial_epoch, show_summary=args.summary)
