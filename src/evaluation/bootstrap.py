"""
src/evaluation/bootstrap.py

Bootstrap confidence intervals for verification metrics.

The bootstrap resamples genuine and impostor score arrays with replacement.
For system differences, the same resampled indices are applied to each system
so ArcFace-vs-Softmax intervals are paired on the same verification pairs.
"""

import numpy as np

from src.utils.metrics import compute_eer, compute_tar_at_far


def _ci(values, ci=95.0):
    alpha = (100.0 - ci) / 2.0
    return [
        float(np.percentile(values, alpha)),
        float(np.percentile(values, 100.0 - alpha)),
    ]


def _metrics_for_scores(gen, imp, num_thresholds):
    eer, _ = compute_eer(gen, imp, num_thresholds=num_thresholds)
    tar_1, _ = compute_tar_at_far(gen, imp, target_far=0.01,
                                  num_thresholds=num_thresholds)
    tar_01, _ = compute_tar_at_far(gen, imp, target_far=0.001,
                                   num_thresholds=num_thresholds)
    return {
        "EER": float(eer),
        "TAR_at_FAR_1pct": float(tar_1),
        "TAR_at_FAR_0.1pct": float(tar_01),
    }


def bootstrap_metric_cis(results: dict, n_boot: int = 500, ci: float = 95.0,
                         seed: int = 42, num_thresholds: int = 1000,
                         max_impostor_per_bootstrap: int = 200000) -> dict:
    """Compute bootstrap CIs for metrics and ArcFace-Softmax differences.

    Args:
        results: mapping system name -> (genuine_scores, impostor_scores).
        n_boot: number of bootstrap resamples.
        ci: confidence interval width in percent.
        seed: random seed for reproducibility.
        num_thresholds: threshold sweep resolution per bootstrap sample.
        max_impostor_per_bootstrap: cap impostor resample size to keep runtime
            bounded for large open-set runs. Genuine scores are always sampled
            at their full count.

    Returns:
        JSON-serialisable dict with per-system metric CIs and paired
        ArcFace-minus-Softmax difference CIs when both systems are present.
    """
    if n_boot <= 0:
        return {}

    rng = np.random.RandomState(seed)
    system_names = list(results.keys())
    n_gen = len(next(iter(results.values()))[0])
    n_imp_full = len(next(iter(results.values()))[1])
    n_imp = min(n_imp_full, max_impostor_per_bootstrap)

    samples = {
        name: {
            "EER": [],
            "TAR_at_FAR_1pct": [],
            "TAR_at_FAR_0.1pct": [],
        }
        for name in system_names
    }
    diffs = {
        "EER": [],
        "TAR_at_FAR_1pct": [],
        "TAR_at_FAR_0.1pct": [],
    }

    for _ in range(n_boot):
        gen_idx = rng.randint(0, n_gen, size=n_gen)
        imp_idx = rng.randint(0, n_imp_full, size=n_imp)

        boot_metrics = {}
        for name, (gen, imp) in results.items():
            metrics = _metrics_for_scores(
                gen[gen_idx],
                imp[imp_idx],
                num_thresholds=num_thresholds,
            )
            boot_metrics[name] = metrics
            for key, value in metrics.items():
                samples[name][key].append(value)

        if "ArcFace" in boot_metrics and "Softmax" in boot_metrics:
            for key in diffs:
                diffs[key].append(boot_metrics["ArcFace"][key]
                                  - boot_metrics["Softmax"][key])

    summary = {
        "n_boot": int(n_boot),
        "ci": float(ci),
        "seed": int(seed),
        "num_thresholds": int(num_thresholds),
        "max_impostor_per_bootstrap": int(max_impostor_per_bootstrap),
        "systems": {},
    }
    for name, values_by_metric in samples.items():
        summary["systems"][name] = {
            key: {
                "mean": float(np.mean(values)),
                "ci": _ci(values, ci=ci),
            }
            for key, values in values_by_metric.items()
        }

    if diffs["EER"]:
        summary["ArcFace_minus_Softmax"] = {
            key: {
                "mean": float(np.mean(values)),
                "ci": _ci(values, ci=ci),
            }
            for key, values in diffs.items()
        }
    return summary
