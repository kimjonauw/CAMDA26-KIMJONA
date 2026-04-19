"""
report_generator.py
====================
Run after attack_synth_shadow.py completes.
Produces:
  - reports/report_<dataset>_<timestamp>.md   ← human-readable
  - reports/report_<dataset>_<timestamp>.json ← machine-readable

Usage (standalone from repo root):
    python report_generator.py --log MIA_on_bulkRNAseq_CAMDA2026/log_combined.txt --dataset COMBINED
    python report_generator.py --log MIA_on_bulkRNAseq_CAMDA2026/run_log.txt --dataset BRCA

Or import and call directly from your pipeline:
    from report_generator import generate_report
    generate_report(dataset_name, trained_results, split_summary)
"""

import os
import re
import sys
import json
import argparse
import numpy as np
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Core report dataclass
# ──────────────────────────────────────────────────────────────────────────────

class ModelReport:
    def __init__(self, clf_name):
        self.clf_name   = clf_name.upper()
        self.val_tpr    = None
        self.val_auc    = None
        self.t_subset   = None
        self.n_noise    = None
        self.split_tprs = {}   # {split_idx: tpr}
        self.split_aucs = {}   # {split_idx: auc}
        self.split_accs = {}   # {split_idx: acc}
        self.ensemble_weight = None

    @property
    def mean_tpr(self):
        v = list(self.split_tprs.values())
        return float(np.mean(v)) if v else None

    @property
    def mean_auc(self):
        v = list(self.split_aucs.values())
        return float(np.mean(v)) if v else None

    @property
    def std_tpr(self):
        v = list(self.split_tprs.values())
        return float(np.std(v)) if len(v) > 1 else 0.0

    def to_dict(self):
        return {
            "clf":            self.clf_name,
            "val_tpr":        self.val_tpr,
            "val_auc":        self.val_auc,
            "mean_split_tpr": self.mean_tpr,
            "std_split_tpr":  self.std_tpr,
            "mean_split_auc": self.mean_auc,
            "t_subset":       self.t_subset,
            "n_noise":        self.n_noise,
            "ensemble_weight":self.ensemble_weight,
            "splits": {
                str(s): {
                    "tpr": self.split_tprs.get(s),
                    "auc": self.split_aucs.get(s),
                    "acc": self.split_accs.get(s),
                }
                for s in sorted(self.split_tprs)
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Log parser  (reads log_combined.txt / run_log.txt)
# ──────────────────────────────────────────────────────────────────────────────

_SPLIT_LINE = re.compile(
    r"\[(?P<clf>\w+)\s*\]\s+Split\s+(?P<split>\d+):\s+"
    r"ACC=(?P<acc>[\d.]+)\s+TPR@10%FPR=(?P<tpr>[\d.]+)\s+AUC=(?P<auc>[\d.]+)"
)
_TABLE_LINE = re.compile(
    r"^\s+(?P<clf>\w+)\s+"
    r"(?P<val_tpr>[\d.]+)\s+(?P<val_auc>[\d.]+)\s+"
    r"(?P<rest>.+?)\s+\[(?P<tsubset>[^\]]+)\]\s*$"
)
_ENSEMBLE_WEIGHT = re.compile(
    r"\[ENSEMBLE\]\s+(?P<clf>\w+)\s+weight=(?P<w>[\d.]+)"
)
_PREDICT_LINE = re.compile(
    r"\[(?P<clf>\w+)\]\s+t_indices=\[(?P<tidx>[^\]]+)\]\s+"
    r"n_noise=(?P<nnoise>\d+)\s+val_tpr=(?P<vtpr>[\d.]+)"
)


def parse_log(log_path: str) -> dict[str, ModelReport]:
    """Parse a run_log.txt or log_combined.txt into ModelReport objects."""
    reports: dict[str, ModelReport] = {}

    def _get(name):
        n = name.upper()
        if n not in reports:
            reports[n] = ModelReport(n)
        return reports[n]

    with open(log_path) as f:
        for line in f:
            # Per-split evaluation lines
            m = _SPLIT_LINE.search(line)
            if m:
                r = _get(m.group("clf"))
                s = int(m.group("split"))
                r.split_tprs[s] = float(m.group("tpr"))
                r.split_aucs[s] = float(m.group("auc"))
                r.split_accs[s] = float(m.group("acc"))
                continue

            # Step-5 prediction lines  →  capture n_noise and val_tpr
            m = _PREDICT_LINE.search(line)
            if m:
                r = _get(m.group("clf"))
                r.n_noise = int(m.group("nnoise"))
                r.val_tpr = float(m.group("vtpr"))
                continue

            # Ensemble weight lines
            m = _ENSEMBLE_WEIGHT.search(line)
            if m:
                r = _get(m.group("clf"))
                r.ensemble_weight = float(m.group("w"))
                continue

            # Summary table line (val_tpr, val_auc, T-subset)
            m = _TABLE_LINE.match(line)
            if m:
                r = _get(m.group("clf"))
                r.val_tpr  = float(m.group("val_tpr"))
                r.val_auc  = float(m.group("val_auc"))
                raw_ts     = m.group("tsubset")
                # Parse something like "1, 2, 5, 150, 200, 300, 500, 750"
                try:
                    r.t_subset = [int(x.strip()) for x in raw_ts.split(",")]
                except ValueError:
                    r.t_subset = raw_ts

    return reports


# ──────────────────────────────────────────────────────────────────────────────
# Direct dict ingestion  (call from pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def reports_from_pipeline(trained_results: dict, split_summary: dict) -> dict[str, ModelReport]:
    """
    trained_results: { clf_name: (clf, history, t_indices, noise_budget, val_tpr, val_auc, selected_ts) }
    split_summary:   { clf_name: { split_idx: {"tpr": ..., "auc": ...} } }
    """
    from mia.config import T_SUPERSET  # only needed when called from pipeline

    reports = {}
    total_w = sum(t[4] for t in trained_results.values()) or 1.0

    for clf_name, entry in trained_results.items():
        _, _, t_indices, noise_budget, val_tpr, val_auc, selected_ts = entry
        r = ModelReport(clf_name)
        r.val_tpr = float(val_tpr)
        r.val_auc = float(val_auc)
        r.t_subset = selected_ts
        r.n_noise  = int(noise_budget)
        r.ensemble_weight = float(val_tpr) / total_w

        for s, metrics in split_summary.get(clf_name, {}).items():
            r.split_tprs[s] = metrics["tpr"]
            r.split_aucs[s] = metrics["auc"]
        reports[clf_name.upper()] = r

    return reports


# ──────────────────────────────────────────────────────────────────────────────
# Markdown renderer
# ──────────────────────────────────────────────────────────────────────────────

def _rank(reports: dict, key: str) -> str:
    """Return clf name ranked #1 by key (mean_tpr or mean_auc)."""
    return max(reports.values(), key=lambda r: getattr(r, key) or 0).clf_name


def render_markdown(dataset: str, reports: dict[str, ModelReport], extra_fpr_rows: dict = None) -> str:
    """Render a compact Markdown report. extra_fpr_rows is optional dict from compute_multi_fpr."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_splits = max((max(r.split_tprs.keys(), default=0) for r in reports.values()), default=0)

    lines = [
        f"# MIA Report — {dataset}",
        f"_Generated: {ts}_\n",
        "## Summary",
        "",
        "| CLF | val_TPR | val_AUC | mean_TPR±std | mean_AUC | T-subset | n_noise | ens_w |",
        "|-----|---------|---------|--------------|----------|----------|---------|-------|",
    ]

    for r in sorted(reports.values(), key=lambda x: x.mean_tpr or 0, reverse=True):
        val_tpr = f"{r.val_tpr:.4f}" if r.val_tpr is not None else "—"
        val_auc = f"{r.val_auc:.4f}" if r.val_auc is not None else "—"
        m_tpr   = f"{r.mean_tpr:.4f}±{r.std_tpr:.4f}" if r.mean_tpr is not None else "—"
        m_auc   = f"{r.mean_auc:.4f}" if r.mean_auc is not None else "—"
        ts_str  = str(r.t_subset) if r.t_subset else "—"
        nn_str  = str(r.n_noise) if r.n_noise else "—"
        ew_str  = f"{r.ensemble_weight:.4f}" if r.ensemble_weight is not None else "—"
        lines.append(f"| {r.clf_name} | {val_tpr} | {val_auc} | {m_tpr} | {m_auc} | {ts_str} | {nn_str} | {ew_str} |")

    # Per-split breakdown
    lines += ["\n## Per-Split TPR@10%FPR\n", "| CLF | " + " | ".join(f"S{s}" for s in range(1, n_splits + 1)) + " |",
              "|-----|" + "------|" * n_splits]
    for r in sorted(reports.values(), key=lambda x: x.mean_tpr or 0, reverse=True):
        cells = " | ".join(f"{r.split_tprs.get(s, 0):.4f}" for s in range(1, n_splits + 1))
        lines.append(f"| {r.clf_name} | {cells} |")

    # Per-split AUC
    lines += ["\n## Per-Split AUC\n", "| CLF | " + " | ".join(f"S{s}" for s in range(1, n_splits + 1)) + " |",
              "|-----|" + "------|" * n_splits]
    for r in sorted(reports.values(), key=lambda x: x.mean_auc or 0, reverse=True):
        cells = " | ".join(f"{r.split_aucs.get(s, 0):.4f}" for s in range(1, n_splits + 1))
        lines.append(f"| {r.clf_name} | {cells} |")

    # Multi-FPR table (if provided)
    if extra_fpr_rows:
        fpr_levels = sorted({fpr for row in extra_fpr_rows.values() for fpr in row})
        lines += ["\n## TPR at Multiple FPR Levels\n",
                  "| CLF | " + " | ".join(f"TPR@{int(f*100)}%FPR" for f in fpr_levels) + " |",
                  "|-----|" + "--------|" * len(fpr_levels)]
        for clf_name, row in sorted(extra_fpr_rows.items()):
            cells = " | ".join(f"{row.get(fpr, 0):.4f}" for fpr in fpr_levels)
            lines.append(f"| {clf_name} | {cells} |")

    # Key insights
    best_tpr_clf = _rank(reports, "mean_tpr")
    best_auc_clf = _rank(reports, "mean_auc")
    lines += [
        "\n## Key Insights",
        f"- **Best TPR@10%FPR**: {best_tpr_clf} ({reports[best_tpr_clf].mean_tpr:.4f} mean over splits)",
        f"- **Best AUC**: {best_auc_clf} ({reports[best_auc_clf].mean_auc:.4f} mean over splits)",
    ]

    # Flag LGBM imbalance issue automatically
    if "LGBM" in reports:
        lgbm = reports["LGBM"]
        avg_acc = np.mean(list(lgbm.split_accs.values())) if lgbm.split_accs else 0
        if avg_acc > 0.79 and (lgbm.mean_tpr or 1) < 0.40:
            lines.append(
                f"- ⚠️  **LGBM** high ACC ({avg_acc:.3f}) but low TPR ({lgbm.mean_tpr:.4f}) — "
                f"likely predicting majority class. Consider `scale_pos_weight` or removing from ensemble."
            )

    val_vs_test = []
    for r in reports.values():
        if r.val_tpr and r.mean_tpr:
            delta = r.mean_tpr - r.val_tpr
            if delta > 0.05:
                val_vs_test.append(f"{r.clf_name} (+{delta:.3f})")
    if val_vs_test:
        lines.append(f"- 📈 Val→Test TPR gain: {', '.join(val_vs_test)} — val proxy may be harder than test splits")

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Multi-FPR computation  (needs raw scores — call from pipeline or pass CSV dir)
# ──────────────────────────────────────────────────────────────────────────────

def tpr_at_fpr(y_true, scores, fpr_threshold):
    """TPR at a given FPR threshold (mirrors classifier._tpr_at_fpr)."""
    from sklearn.metrics import roc_curve
    y_true = np.array(y_true, dtype=int)
    scores = np.array(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fprs, tprs, _ = roc_curve(y_true, scores)
    # Find the largest TPR where FPR ≤ threshold
    valid = fprs <= fpr_threshold + 1e-9
    return float(tprs[valid][-1]) if valid.any() else 0.0


def compute_multi_fpr(
    trained_results: dict,
    split_features: dict,  # { split_idx: (raw_features, y_member) }
    fpr_levels=(0.01, 0.05, 0.10, 0.20),
) -> dict[str, dict[float, float]]:
    """
    Compute TPR at multiple FPR levels averaged over all splits.

    trained_results : from step_train_classifiers
    split_features  : { split_idx: (raw_superset_array, y_member_array) }
    fpr_levels      : iterable of FPR thresholds (fractions, e.g. 0.01 for 1%)

    Returns: { clf_name: { fpr: mean_tpr_over_splits } }
    """
    try:
        from mia.classifier import get_classifier
        from mia.loss_features import slice_raw_features, summarize_features
    except ImportError:
        raise ImportError("Must be run from inside MIA_on_bulkRNAseq_CAMDA2026/ or with mia on PYTHONPATH")

    results = {}
    for clf_name, entry in trained_results.items():
        clf, _, t_indices, noise_budget = entry[:4]
        entry_obj = get_classifier(clf_name)
        split_tprs = {fpr: [] for fpr in fpr_levels}

        for s, (raw, y_member) in split_features.items():
            X_feat = summarize_features(
                slice_raw_features(raw, t_indices, noise_budget),
                noise_budget, len(t_indices)
            )
            scores = entry_obj.predict(clf, X_feat)
            for fpr in fpr_levels:
                split_tprs[fpr].append(tpr_at_fpr(y_member, scores, fpr))

        results[clf_name.upper()] = {fpr: float(np.mean(v)) for fpr, v in split_tprs.items()}

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main  (standalone log-file mode)
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(
    dataset: str,
    trained_results: dict = None,
    split_summary: dict = None,
    log_path: str = None,
    extra_fpr_rows: dict = None,
    out_dir: str = "reports",
):
    """
    Entry point for both pipeline-integrated and standalone log-parse modes.

    Either pass trained_results + split_summary (from pipeline),
    or pass log_path (standalone).
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if log_path:
        reports = parse_log(log_path)
    elif trained_results and split_summary:
        reports = reports_from_pipeline(trained_results, split_summary)
    else:
        raise ValueError("Provide either log_path or (trained_results + split_summary)")

    if not reports:
        print("[report_generator] WARNING: no model data found — check log format or inputs")
        return

    md   = render_markdown(dataset, reports, extra_fpr_rows=extra_fpr_rows)
    data = {clf: r.to_dict() for clf, r in reports.items()}

    md_path   = os.path.join(out_dir, f"report_{dataset}_{stamp}.md")
    json_path = os.path.join(out_dir, f"report_{dataset}_{stamp}.json")

    with open(md_path,   "w") as f: f.write(md)
    with open(json_path, "w") as f: json.dump(data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  REPORT SAVED")
    print(f"  Markdown : {md_path}")
    print(f"  JSON     : {json_path}")
    print(f"{'='*60}\n")
    print(md)
    return md_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate condensed MIA report from log file")
    parser.add_argument("--log",     required=True,  help="Path to run_log.txt or log_combined.txt")
    parser.add_argument("--dataset", required=True,  help="Dataset name (BRCA, COMBINED, etc.)")
    parser.add_argument("--out-dir", default="reports", help="Output directory (default: reports/)")
    args = parser.parse_args()

    generate_report(dataset=args.dataset, log_path=args.log, out_dir=args.out_dir)
