"""
src/models/train_arcface.py

ArcFace training script for IrisNet.

Combines the IrisNet backbone (which outputs L2-normalised 512-D embeddings)
with the ArcFaceLayer classification head.  At inference time only the
backbone is needed — embeddings are compared via cosine similarity.

Training detail
---------------
ArcFace requires the one-hot labels during the forward pass (so it can apply
the angular margin only to the true-class angle).  This means the model has
TWO inputs: [image, label_onehot].  A thin wrapper model is built to fuse
them for Keras's model.fit() API.

The training uses:
  - SGD with momentum (0.9) and weight decay (5e-4) instead of Adam, to
    prevent the backbone from collapsing while the W-matrix absorbs all
    discriminative capacity.
  - Margin/scale annealing: warmup with m=0 s=16, then linear ramp to
    m=0.5 s=64 over RAMPUP_EPOCHS.
  - Deterministic step LR decay at epochs 40/60/80 (standard ArcFace schedule).

Saved artefacts
---------------
  models/arcface_best.h5             Best full-model checkpoint (backbone + head)
  models/arcface_backbone.weights.h5 Backbone-only weights (used for inference)
  models/arcface_history.json        Training history

Usage
-----
    python -m src.models.train_arcface
    python -m src.models.train_arcface --epochs 100 --batch_size 64
    python -m src.models.train_arcface --cpu        # force CPU (disables DirectML)
"""

import argparse
import json
import os

import tensorflow as tf

from src.models.architecture import build_irisnet
from src.models.arcface_loss import ArcFaceLayer
from src.utils.data_loader import build_datasets

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS         = 50
BATCH_SIZE     = 64
LR_INITIAL     = 0.01      # SGD base LR
EMBEDDING_DIM  = 512
ARCFACE_MARGIN = 0.5       # target margin (annealed from 0.0)
ARCFACE_SCALE  = 64.0      # target scale  (annealed from 16.0)
WARMUP_EPOCHS  = 5         # softmax-only warmup (m=0, s=16)
RAMPUP_EPOCHS  = 15        # linear ramp m: 0→0.5, s: 16→64
MIN_SAMPLES    = 2         # exclude single-sample classes

CHECKPOINT_PATH         = 'models/arcface_best.h5'
LATEST_PATH             = 'models/arcface_latest.h5'
BACKBONE_PATH           = 'models/arcface_backbone.weights.h5'
HISTORY_PATH            = 'models/arcface_history.json'
OPENSET_CHECKPOINT_PATH = 'models/arcface_openset_best.h5'
OPENSET_LATEST_PATH     = 'models/arcface_openset_latest.h5'
OPENSET_BACKBONE_PATH   = 'models/arcface_openset_backbone.weights.h5'
OPENSET_HISTORY_PATH    = 'models/arcface_openset_history.json'
EXPANDED_OPENSET_CHECKPOINT_PATH = 'models/arcface_expanded_openset_best.h5'
EXPANDED_OPENSET_LATEST_PATH     = 'models/arcface_expanded_openset_latest.h5'
EXPANDED_OPENSET_BACKBONE_PATH   = 'models/arcface_expanded_openset_backbone.weights.h5'
EXPANDED_OPENSET_HISTORY_PATH    = 'models/arcface_expanded_openset_history.json'
EXPANDED_OPENSET_SPLIT_PATH      = 'data/test_split_expanded_openset.json'
EXPANDED_ROOTS = ['data/processed', 'data/processed_recovered']


# ── Margin / Scale Annealing ─────────────────────────────────────────────────

class MarginScaleAnnealingCallback(tf.keras.callbacks.Callback):
    """Anneal ArcFace margin and scale during training.

    Schedule:
      - Warmup  (epoch 0 .. warmup-1):  m = 0.0,           s = initial_scale
      - Ramp-up (warmup .. warmup+ramp): m linearly → target_m, s linearly → target_s
      - Full    (after ramp):            m = target_m,      s = target_s
    """

    def __init__(self, warmup_epochs, rampup_epochs,
                 target_margin, target_scale, initial_scale=16.0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.target_margin = target_margin
        self.target_scale = target_scale
        self.initial_scale = initial_scale

    def on_epoch_begin(self, epoch, logs=None):
        arcface = self.model.get_layer('arcface')
        if epoch < self.warmup_epochs:
            m, s = 0.0, self.initial_scale
        elif epoch < self.warmup_epochs + self.rampup_epochs:
            progress = (epoch - self.warmup_epochs) / self.rampup_epochs
            m = self.target_margin * progress
            s = self.initial_scale + (self.target_scale - self.initial_scale) * progress
        else:
            m, s = self.target_margin, self.target_scale

        arcface.margin_var.assign(m)
        arcface.scale_var.assign(s)
        if epoch < self.warmup_epochs + self.rampup_epochs + 1 or epoch % 10 == 0:
            print(f'  [anneal] epoch {epoch}: margin={m:.3f}, scale={s:.1f}')


def build_arcface_model(num_classes: int, num_train_samples: int,
                        batch_size: int = BATCH_SIZE,
                        start_epoch: int = 0,
                        embedding_dim: int = EMBEDDING_DIM):
    """Build and compile the full ArcFace training model.

    Architecture (training):
        image (128,128,1)  ─┐
                             ├─ IrisNet backbone ─ ArcFaceLayer ─ scaled logits
        label_onehot        ─┘

    Args:
        num_classes:      number of identity classes
        num_train_samples: number of training samples (for LR schedule)
        embedding_dim:    IrisNet embedding dimension

    Returns:
        (training_model, backbone)
          training_model: compiled model with two inputs [image, label_onehot]
          backbone:       IrisNet base model (single image → embedding)
    """
    backbone = build_irisnet(input_shape=(128, 128, 1), embedding_dim=embedding_dim)

    # Two inputs for the training wrapper
    img_input   = tf.keras.Input(shape=(128, 128, 1),    name='image')
    label_input = tf.keras.Input(shape=(num_classes,),   name='label_onehot',
                                 dtype=tf.float32)

    embeddings = backbone(img_input, training=True)

    # Start with m=0, s=16 — annealed by MarginScaleAnnealingCallback
    arcface_layer = ArcFaceLayer(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=0.0,
        scale=16.0,
        name='arcface',
    )
    logits = arcface_layer([embeddings, label_input])

    training_model = tf.keras.Model(
        inputs=[img_input, label_input],
        outputs=logits,
        name='IrisNet_ArcFace',
    )

    # Step LR decay at epochs 25/35/45 (scaled for 50-epoch schedule)
    steps_per_epoch = num_train_samples // batch_size + 1
    lr_boundaries = [25, 35, 45]
    lr_values = [LR_INITIAL, LR_INITIAL * 0.1,
                 LR_INITIAL * 0.01, LR_INITIAL * 0.001]
    current_stage = sum(start_epoch >= b for b in lr_boundaries)
    future_boundaries = [
        steps_per_epoch * (b - start_epoch)
        for b in lr_boundaries
        if b > start_epoch
    ]
    if future_boundaries:
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=future_boundaries,
            values=lr_values[current_stage:],
        )
    else:
        lr_schedule = lr_values[current_stage]

    training_model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=0.9,
            weight_decay=5e-4,
        ),
        # ArcFace outputs raw scaled logits → from_logits=True
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        jit_compile=False,
    )
    return training_model, backbone


def _adapt_dataset_for_arcface(ds: tf.data.Dataset):
    """Re-map a (image, label_onehot) dataset to ([image, label_onehot], label_onehot).

    Keras model.fit() expects (inputs, targets).  Because the ArcFace training
    model has two inputs, the input must be [image, label_onehot] while the
    target is the same label_onehot (for cross-entropy).
    """
    return ds.map(
        lambda x, y: ({'image': x, 'label_onehot': y}, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


class EmbeddingDiversityCallback(tf.keras.callbacks.Callback):
    """Monitor embedding spread to catch ArcFace collapse early."""

    def __init__(self, backbone, val_ds, batches=4):
        super().__init__()
        self.backbone = backbone
        self.val_ds = val_ds
        self.batches = batches

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        embeddings = []
        for i, (x, _) in enumerate(self.val_ds):
            if i >= self.batches:
                break
            embeddings.append(self.backbone(x, training=False).numpy())
        if not embeddings:
            return
        emb = tf.concat([tf.convert_to_tensor(e) for e in embeddings], axis=0).numpy()
        mean_std = float(emb.std(axis=0).mean())
        logs['val_embedding_std'] = mean_std
        print(f'  [embedding] epoch {epoch}: val_embedding_std={mean_std:.6f}')


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
                  backbone=None, val_ds=None,
                  initial_value_threshold=None):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    callbacks = [
        MarginScaleAnnealingCallback(
            warmup_epochs=WARMUP_EPOCHS,
            rampup_epochs=RAMPUP_EPOCHS,
            target_margin=ARCFACE_MARGIN,
            target_scale=ARCFACE_SCALE,
            initial_scale=16.0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            initial_value_threshold=initial_value_threshold,
            verbose=1,
        ),
        # No EarlyStopping — the margin/scale ramp-up causes val_accuracy
        # to drop temporarily, which misleads patience-based stopping.
        # Instead, rely on the full 100-epoch schedule with LR step decay
        # at epochs 40/60/80 (standard ArcFace protocol).
    ]
    if backbone is not None and val_ds is not None:
        callbacks.append(EmbeddingDiversityCallback(backbone, val_ds))
    return callbacks


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, cpu: bool = False,
          openset: bool = False, expanded: bool = False,
          processed_roots: list = None, resume: bool = False,
          initial_epoch: int = 0, show_summary: bool = False):
    if cpu:
        tf.config.set_visible_devices([], 'GPU')
        print('[train_arcface] GPU disabled — running on CPU')

    if expanded and not openset:
        raise ValueError('Expanded training is currently defined for --openset only')
    if expanded:
        checkpoint_path = EXPANDED_OPENSET_CHECKPOINT_PATH
        latest_path = EXPANDED_OPENSET_LATEST_PATH
        backbone_path = EXPANDED_OPENSET_BACKBONE_PATH
        history_path = EXPANDED_OPENSET_HISTORY_PATH
        split_path = EXPANDED_OPENSET_SPLIT_PATH
        default_roots = EXPANDED_ROOTS
    else:
        checkpoint_path = OPENSET_CHECKPOINT_PATH if openset else CHECKPOINT_PATH
        latest_path = OPENSET_LATEST_PATH if openset else LATEST_PATH
        backbone_path = OPENSET_BACKBONE_PATH if openset else BACKBONE_PATH
        history_path = OPENSET_HISTORY_PATH if openset else HISTORY_PATH
        split_path = None
        default_roots = ['data/processed']
    processed_root = processed_roots if processed_roots else default_roots
    split_mode = 'identity_disjoint' if openset else 'stratified'

    print('=' * 60)
    mode = 'expanded open-set' if expanded else ('open-set' if openset else 'closed-set')
    print(f'IrisNet — ArcFace Training  ({mode})')
    print('=' * 60)
    print(f'Processed roots: {processed_root}')

    train_ds, val_ds, _, num_classes = build_datasets(
        processed_root=processed_root, batch_size=batch_size,
        min_samples=MIN_SAMPLES, split_mode=split_mode,
        test_split_path=split_path,
    )
    # Count training samples for LR schedule boundaries
    num_train_samples = sum(1 for _ in train_ds.unbatch())
    print(f'Classes: {num_classes}  |  Batch size: {batch_size}  |  Epochs: {epochs}')
    print(f'Train samples: {num_train_samples}  |  Min samples/class: {MIN_SAMPLES}')
    print(f'ArcFace target: margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE}')
    print(f'Warmup: {WARMUP_EPOCHS} epochs (m=0, s=16)  |  Ramp-up: {RAMPUP_EPOCHS} epochs')
    print(f'Optimizer: SGD(lr={LR_INITIAL}, momentum=0.9, wd=5e-4)')

    # Wrap datasets so both inputs and targets are provided
    train_ds_af = _adapt_dataset_for_arcface(train_ds)
    val_ds_af   = _adapt_dataset_for_arcface(val_ds)

    training_model, backbone = build_arcface_model(
        num_classes, num_train_samples=num_train_samples,
        batch_size=batch_size, start_epoch=initial_epoch,
    )
    if resume:
        resume_path = latest_path if os.path.isfile(latest_path) else checkpoint_path
        if os.path.isfile(resume_path):
            training_model.load_weights(resume_path)
            print(f'[train_arcface] Resumed weights from {resume_path}')
    if show_summary:
        training_model.summary()

    previous_history = _load_history(history_path) if resume else {}
    checkpoint_threshold = _best_history_value(previous_history, 'val_accuracy', 'max')
    if checkpoint_threshold is not None:
        print(f'[train_arcface] Checkpoint resumes from best val_accuracy={checkpoint_threshold:.6f}')

    history = training_model.fit(
        train_ds_af,
        validation_data=val_ds_af,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=get_callbacks(
            checkpoint_path,
            backbone=backbone,
            val_ds=val_ds,
            initial_value_threshold=checkpoint_threshold,
        ),
        verbose=2,
    )

    training_model.save(latest_path)
    print(f'Latest full model saved -> {latest_path}')

    # Save backbone separately for Phase 6 inference
    backbone.save_weights(backbone_path)
    print(f'Backbone weights saved -> {backbone_path}')

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    merged_history = _merge_history(previous_history, history.history, initial_epoch)
    with open(history_path, 'w') as f:
        json.dump(merged_history, f, indent=2)
    print(f'History saved -> {history_path}')
    print(f'Best full model saved -> {checkpoint_path}')
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
