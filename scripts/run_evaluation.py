"""
scripts/run_evaluation.py

Standalone evaluation runner for Phase 6 (closed-set) and Phase 7 (open-set).

Loads the test split, extracts embeddings from all three systems, computes
similarity scores, and produces:
  - A results JSON with EER, TAR@FAR metrics and score summaries
  - All five evaluation plots (ROC, DET, score distributions, t-SNE, training
    curves) saved to figures/ (closed-set) or figures/openset/ (open-set)

Usage
-----
    python -m scripts.run_evaluation                # closed-set
    python -m scripts.run_evaluation --openset      # open-set
"""

import argparse
import json
import os

# Disable XLA autotuner (crashes on RTX 5090 Blackwell)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# Set LD_LIBRARY_PATH for pip-installed NVIDIA libraries
_NVIDIA_LIB = os.path.abspath('.venv/lib/python3.11/site-packages/nvidia')
if os.path.isdir(_NVIDIA_LIB):
    _lib_dirs = [
        f'{_NVIDIA_LIB}/cudnn/lib', f'{_NVIDIA_LIB}/cublas/lib',
        f'{_NVIDIA_LIB}/cuda_runtime/lib', f'{_NVIDIA_LIB}/cufft/lib',
        f'{_NVIDIA_LIB}/cusparse/lib', f'{_NVIDIA_LIB}/cusolver/lib',
        f'{_NVIDIA_LIB}/nvjitlink/lib', f'{_NVIDIA_LIB}/cuda_cupti/lib',
        f'{_NVIDIA_LIB}/cuda_nvrtc/lib', f'{_NVIDIA_LIB}/nccl/lib',
        f'{_NVIDIA_LIB}/curand/lib',
    ]
    os.environ['LD_LIBRARY_PATH'] = (':'.join(_lib_dirs) + ':'
                                     + os.environ.get('LD_LIBRARY_PATH', ''))

import numpy as np
import matplotlib.pyplot as plt

import src.evaluation.embeddings as emb_mod
from src.evaluation.embeddings import (
    load_test_images, extract_arcface_embeddings,
    extract_softmax_embeddings, extract_gabor_codes,
)
from src.evaluation.pairs import (
    generate_pairs, compute_cosine_scores, compute_hamming_scores,
)
from src.evaluation.plotting import (
    build_comparison_table, plot_roc_curves, plot_det_curves,
    plot_score_distributions, plot_training_curves, plot_embedding_tsne,
)
from src.evaluation.bootstrap import bootstrap_metric_cis
from src.utils.data_loader import load_test_split
from src.utils.metrics import compute_eer, compute_tar_at_far


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {path}')


def _infer_softmax_num_classes(h5_path: str) -> int:
    """Read the softmax_head weight shape from an h5 checkpoint.

    The trailing Dense layer has kernel shape (512, num_classes); this avoids
    hardcoding class counts per training run.
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        for grp in ('model_weights/softmax_head/softmax_head',
                    'model_weights/softmax_head'):
            if grp in f:
                for key in ('kernel', 'kernel:0'):
                    if key in f[grp]:
                        return int(f[grp][key].shape[1])
    raise RuntimeError(f'Could not infer num_classes from {h5_path}')


def _infer_arcface_num_classes(h5_path: str) -> int:
    """Read the ArcFace W matrix shape (embedding_dim, num_classes)."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        for grp in ('model_weights/arcface/arcface',
                    'model_weights/arcface'):
            if grp in f:
                for key in ('arcface_weights', 'W', 'W:0', 'kernel'):
                    if key in f[grp]:
                        return int(f[grp][key].shape[1])
    raise RuntimeError(f'Could not infer num_classes from {h5_path}')


def _select_paths(openset: bool, expanded: bool):
    if expanded and not openset:
        raise ValueError('--expanded evaluation is defined for --openset only')
    if expanded:
        return {
            'suffix': 'expanded-openset',
            'split_path': 'data/test_split_expanded_openset.json',
            'arcface_h5': 'models/arcface_expanded_openset_best.h5',
            'softmax_h5': 'models/softmax_expanded_openset_best.h5',
            'arcface_hist': 'models/arcface_expanded_openset_history.json',
            'softmax_hist': 'models/softmax_expanded_openset_history.json',
            'figures_dir': 'figures/expanded_openset',
            'results_json': 'reports/phase8_expanded_openset_results.json',
        }
    return {
        'suffix': 'openset' if openset else 'closed-set',
        'split_path': 'data/test_split_openset.json' if openset else 'data/test_split.json',
        'arcface_h5': 'models/arcface_openset_best.h5' if openset else 'models/arcface_best.h5',
        'softmax_h5': 'models/softmax_openset_best.h5' if openset else 'models/softmax_best.h5',
        'arcface_hist': 'models/arcface_openset_history.json' if openset else 'models/arcface_history.json',
        'softmax_hist': 'models/softmax_openset_history.json' if openset else 'models/softmax_history.json',
        'figures_dir': 'figures/openset' if openset else 'figures',
        'results_json': 'reports/phase7_openset_results.json' if openset else 'reports/phase6_closedset_results.json',
    }


def _load_split_payload(path):
    with open(path) as f:
        return json.load(f)


def _validate_identity_disjoint_split(split_payload):
    train = set(split_payload.get('train_identities', []))
    test = set(split_payload.get('test_identities', []))
    if not train or not test:
        return None
    overlap = sorted(train & test)
    return {
        'train_identities': len(train),
        'test_identities': len(test),
        'overlap_count': len(overlap),
        'is_disjoint': len(overlap) == 0,
    }


def run(openset: bool = False, expanded: bool = False,
        bootstrap_iters: int = 0, bootstrap_thresholds: int = 1000,
        bootstrap_max_impostor: int = 200000,
        arcface_h5_override: str = None,
        results_json_override: str = None,
        skip_plots: bool = False):
    paths_cfg = _select_paths(openset, expanded)
    suffix = paths_cfg['suffix']
    split_path = paths_cfg['split_path']
    arcface_h5 = paths_cfg['arcface_h5']
    softmax_h5 = paths_cfg['softmax_h5']
    arcface_hist = paths_cfg['arcface_hist']
    softmax_hist = paths_cfg['softmax_hist']
    figures_dir = paths_cfg['figures_dir']
    results_json = paths_cfg['results_json']
    if arcface_h5_override:
        arcface_h5 = arcface_h5_override
    if results_json_override:
        results_json = results_json_override

    print('=' * 60)
    print(f'Evaluation — {suffix}')
    print('=' * 60)
    if arcface_h5_override:
        print(f'ArcFace checkpoint override: {arcface_h5}')

    # 1. Load test set
    split_payload = _load_split_payload(split_path)
    paths, labels, _ = load_test_split(split_path)
    images = load_test_images(paths)

    # Per-system class counts inferred from each checkpoint (they differ in
    # open-set because Softmax uses min_samples=1 and ArcFace uses min_samples=2).
    n_arc = _infer_arcface_num_classes(arcface_h5)
    n_sm  = _infer_softmax_num_classes(softmax_h5)
    print(f'Test samples: {len(paths)}  |  ArcFace classes: {n_arc}  |  Softmax classes: {n_sm}')

    n_identities = len(set(labels))
    print(f'Test identities: {n_identities}')
    disjoint = _validate_identity_disjoint_split(split_payload)
    if disjoint:
        print(f'Identity disjoint: {disjoint["is_disjoint"]} '
              f'(overlap={disjoint["overlap_count"]})')

    # 2. Generate pairs
    genuine_pairs, impostor_pairs = generate_pairs(labels, seed=42, impostor_ratio=100)

    # 3. Override module-level paths, then extract embeddings/codes
    emb_mod.ARCFACE_MODEL_PATH = arcface_h5
    emb_mod.SOFTMAX_MODEL_PATH = softmax_h5

    print('\n=== ArcFace embeddings ===')
    emb_arcface = extract_arcface_embeddings(images, num_classes=n_arc)

    print('\n=== Softmax embeddings ===')
    emb_softmax = extract_softmax_embeddings(images, num_classes=n_sm)

    print('\n=== Gabor IrisCodes ===')
    codes_gabor = extract_gabor_codes(images)

    # 4. Compute scores
    gen_arc = compute_cosine_scores(emb_arcface, genuine_pairs)
    imp_arc = compute_cosine_scores(emb_arcface, impostor_pairs)
    gen_sm  = compute_cosine_scores(emb_softmax, genuine_pairs)
    imp_sm  = compute_cosine_scores(emb_softmax, impostor_pairs)
    gen_gb  = compute_hamming_scores(codes_gabor, genuine_pairs)
    imp_gb  = compute_hamming_scores(codes_gabor, impostor_pairs)

    results = {
        'ArcFace': (gen_arc, imp_arc),
        'Softmax': (gen_sm,  imp_sm),
        'Gabor':   (gen_gb,  imp_gb),
    }

    # 5. Comparison table + metrics JSON
    table = build_comparison_table(results)
    print('\n' + table.to_string())

    metrics_summary = {'mode': suffix, 'test_samples': len(paths),
                       'test_identities': n_identities,
                       'genuine_pairs': len(genuine_pairs),
                       'impostor_pairs': len(impostor_pairs),
                       'split_path': split_path,
                       'systems': {}}
    if disjoint:
        metrics_summary['identity_disjoint_check'] = disjoint
    for name, (gen, imp) in results.items():
        eer, thr = compute_eer(gen, imp)
        tar_1,  _ = compute_tar_at_far(gen, imp, 0.01)
        tar_01, _ = compute_tar_at_far(gen, imp, 0.001)
        metrics_summary['systems'][name] = {
            'EER': float(eer),
            'EER_threshold':     float(thr),
            'TAR_at_FAR_1pct':   float(tar_1),
            'TAR_at_FAR_0.1pct': float(tar_01),
            'genuine_mean':      float(gen.mean()),
            'impostor_mean':     float(imp.mean()),
            'gap':               float(gen.mean() - imp.mean()),
            'genuine_pairs':     int(len(gen)),
            'impostor_pairs':    int(len(imp)),
        }

    if bootstrap_iters > 0:
        print(f'\n=== Bootstrap CIs ({bootstrap_iters} resamples) ===')
        metrics_summary['bootstrap'] = bootstrap_metric_cis(
            results,
            n_boot=bootstrap_iters,
            num_thresholds=bootstrap_thresholds,
            max_impostor_per_bootstrap=bootstrap_max_impostor,
        )
        diff = metrics_summary['bootstrap'].get('ArcFace_minus_Softmax', {})
        if diff:
            for key, payload in diff.items():
                lo, hi = payload['ci']
                print(f'  ArcFace-Softmax {key}: mean={payload["mean"]:.4f} '
                      f'CI=[{lo:.4f}, {hi:.4f}]')

    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    with open(results_json, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f'\nResults JSON saved -> {results_json}')

    if skip_plots:
        print('\nPlot generation skipped.')
        print('\nDone.')
        return metrics_summary

    # 6. Plots
    print('\n=== Generating plots ===')
    _save_fig(plot_roc_curves(results),           f'{figures_dir}/roc_curves.png')
    _save_fig(plot_det_curves(results),           f'{figures_dir}/det_curves.png')
    _save_fig(plot_score_distributions(results),  f'{figures_dir}/score_distributions.png')

    if os.path.isfile(arcface_hist) and os.path.isfile(softmax_hist):
        _save_fig(
            plot_training_curves({'Softmax': softmax_hist, 'ArcFace': arcface_hist}),
            f'{figures_dir}/training_curves.png',
        )

    labels_arr = np.array(labels)
    _save_fig(
        plot_embedding_tsne(emb_arcface, labels_arr,
                            title=f't-SNE — ArcFace ({suffix}, top 20 identities)',
                            top_k=20),
        f'{figures_dir}/tsne_arcface.png',
    )
    _save_fig(
        plot_embedding_tsne(emb_softmax, labels_arr,
                            title=f't-SNE — Softmax ({suffix}, top 20 identities)',
                            top_k=20),
        f'{figures_dir}/tsne_softmax.png',
    )
    print('\nDone.')
    return metrics_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openset', action='store_true',
                        help='Evaluate the open-set (identity-disjoint) models')
    parser.add_argument('--expanded', action='store_true',
                        help='Evaluate expanded open-set experiment artifacts')
    parser.add_argument('--bootstrap-iters', type=int, default=0,
                        help='Number of bootstrap resamples for metric CIs')
    parser.add_argument('--bootstrap-thresholds', type=int, default=1000,
                        help='Threshold sweep resolution inside bootstrap')
    parser.add_argument('--bootstrap-max-impostor', type=int, default=200000,
                        help='Maximum impostor scores per bootstrap resample')
    parser.add_argument('--arcface-h5', default=None,
                        help='Override ArcFace checkpoint path for comparison runs')
    parser.add_argument('--results-json', default=None,
                        help='Override results JSON output path')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip plot generation for quick comparison runs')
    args = parser.parse_args()
    run(openset=args.openset or args.expanded,
        expanded=args.expanded,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_thresholds=args.bootstrap_thresholds,
        bootstrap_max_impostor=args.bootstrap_max_impostor,
        arcface_h5_override=args.arcface_h5,
        results_json_override=args.results_json,
        skip_plots=args.skip_plots)
