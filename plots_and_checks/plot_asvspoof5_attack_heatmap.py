"""
plot_asvspoof5_attack_heatmap.py

Publication-quality heatmap comparing per-attack detection accuracy across
Spooftral model variants on ASVspoof5 (Development and Evaluation splits).

Usage:
    python plot_asvspoof5_attack_heatmap.py

Edit the MODELS list at the top of the file to change paths / add models.
Set a model's "path" to None to skip it (cells rendered as grey NaN).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# ==========================================================================
# Input file paths — edit here to point to desired checkpoints.
# ==========================================================================

MODELS = [
    {
        "label": "Spooftral\n LLM-only",
        "path": (
            "/path/to/voxtral_clean/"
            "voxtral_clean-generation-asvspoof5-aug-bfloat16-2-constant-"
            "classification_as_generation-with-aug21LLM/"
            "scores_text_outputs/checkpoint-epoch6_eer.txt"
        ),
    },
    {
        "label": "Spooftral",
        "path": (
            "/path/to/"
            "voxtral_clean/voxtral_clean-generation-asvspoof5-aug-bfloat16-2-constant-"
            "classification_as_generation-with-aug-no-margin/"
            "scores_text_outputs/checkpoint-epoch5_eer.txt"
        ),
    },
    {
        "label": "Spooftral\n+ Margin",
        "path": (
            "/path/to/voxtral_clean/"
            "voxtral_clean-generation-asvspoof5-aug-bfloat16-2-constant-"
            "classification_as_generation-with-aug21/"
            "scores_text_outputs/checkpoint-epoch9_eer.txt"
        ),
    },
    {
        "label": "Spooftral-Enc",
        "path": (
            "/path/to/voxtral_clean/"
            "voxtral_clean-classification-head-asvspoof5-bfloat16-dora-2/"
            "scores_text_outputs/checkpoint-epoch5_eer.txt"
        ),
    },
]

DEV_ATTACKS  = ["A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
EVAL_ATTACKS = [
    "A17", "A18", "A19", "A20", "A21", "A22", "A23",
    "A24", "A25", "A26", "A27", "A28", "A29", "A30", "A31", "A32",
]

HIGHLIGHT_DEV  = {"A13", "A15"}
HIGHLIGHT_EVAL = {"A28"}

OUTPUT_PREFIX = "asvspoof5_attack_heatmap"

# ==========================================================================
# 1. Parsing
# ==========================================================================

def _parse_attack_block(lines: list, start: int) -> dict:
    """Read 'AXX | correct / total' lines starting at *start*.

    Stops at: empty line, 'bonafide', 'Codec', or non-pipe line.
    Returns {attack_id: (correct, total)}.
    """
    data = {}
    for ln in lines[start:]:
        stripped = ln.strip()
        if not stripped:
            break
        lower = stripped.lower()
        if lower.startswith("bonafide") or lower.startswith("codec"):
            break
        if "|" not in stripped:
            break
        left, right = stripped.split("|", 1)
        attack_id = left.strip()
        parts = right.split("/")
        if len(parts) != 2:
            print(f"  [WARN] unexpected format: {ln!r} — skipped", file=sys.stderr)
            continue
        try:
            correct = int(parts[0].strip())
            total   = int(parts[1].strip())
        except ValueError:
            print(f"  [WARN] cannot parse ints: {ln!r} — skipped", file=sys.stderr)
            continue
        data[attack_id] = (correct, total)
    return data


def parse_file(path) -> dict:
    """Parse one EER text file.

    Returns:
        {
            'dev':  {attack_id: (correct, total), ...},   # empty if absent
            'eval': {attack_id: (correct, total), ...},   # empty if absent
        }
    Missing file or path=None → both sub-dicts empty (NaN in heatmap).
    """
    empty = {"dev": {}, "eval": {}}

    if path is None:
        return empty

    if not os.path.isfile(path):
        print(f"  [WARN] File not found: {path}", file=sys.stderr)
        return empty

    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    def _find(keyword: str, start: int = 0) -> int:
        for i, ln in enumerate(lines[start:], start):
            # Use 'in' so trailing tags like '[resumed]' don't break matching.
            if keyword in ln:
                return i
        return -1

    result = {"dev": {}, "eval": {}}

    # -- Dev section (may be absent in classification-head files) ------------
    dev_idx = _find("Dev set eval stats")
    if dev_idx != -1:
        col_idx = _find("Attack  | Correct / Total", dev_idx)
        if col_idx != -1:
            result["dev"] = _parse_attack_block(lines, col_idx + 1)

    # -- Eval section --------------------------------------------------------
    eval_idx = _find("Evaluation set eval stats")
    if eval_idx == -1:
        print(f"  [WARN] 'Evaluation set eval stats' not found in:\n    {path}",
              file=sys.stderr)
    else:
        col_idx = _find("Attack  | Correct / Total", eval_idx)
        if col_idx != -1:
            result["eval"] = _parse_attack_block(lines, col_idx + 1)

    return result


# ==========================================================================
# 2. Accuracy computation
# ==========================================================================

def compute_accuracy(raw: dict) -> dict:
    """Convert {split: {attack: (correct, total)}} → {split: {attack: acc%}}.

    NaN is stored where total == 0.
    """
    out = {}
    for split, counts in raw.items():
        acc = {}
        for attack, (correct, total) in counts.items():
            acc[attack] = correct / total * 100.0 if total > 0 else float("nan")
        out[split] = acc
    return out


# ==========================================================================
# 3. Matrix builder
# ==========================================================================

def build_matrix(model_accs: list, attack_list: list, split: str) -> np.ndarray:
    """Build a 2-D float array of shape (n_attacks, n_models).

    Cells are NaN where data is unavailable.
    """
    mat = np.full((len(attack_list), len(model_accs)), np.nan)
    for j, acc in enumerate(model_accs):
        split_acc = acc.get(split, {})
        for i, atk in enumerate(attack_list):
            if atk in split_acc:
                mat[i, j] = split_acc[atk]
    return mat


# ==========================================================================
# 4. Heatmap panel
# ==========================================================================

def plot_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list,
    col_labels: list,
    title: str,
    highlight_rows: set,
    vmin: float,
    vmax: float,
    cmap,
    show_colorbar: bool = False,
    fig=None,
) -> object:
    """Draw one heatmap panel on *ax*.

    Returns the AxesImage (for a shared colorbar if desired).
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_inst = matplotlib.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap_inst.set_bad(color="#BBBBBB")   # grey for NaN

    im = ax.imshow(matrix, cmap=cmap_inst, norm=norm, aspect="auto",
                   interpolation="nearest")

    n_rows, n_cols = matrix.shape

    # White cell dividers
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="white", linewidth=0.9, zorder=2)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="white", linewidth=0.9, zorder=2)

    # Value annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=8.5, color="#999999")
                continue
            # Dark text on light cells, light text on dark cells
            norm_val = (val - vmin) / max(vmax - vmin, 1e-9)
            txt_color = "black" if norm_val > 0.38 else "white"
            ax.text(j, i, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=9.0, color=txt_color)

    # Axes labels
    ax.set_xticks(range(n_cols))
    # Use MultilineText-friendly kwargs: wrap is not needed since labels
    # already contain \n; just set va to "top" so multi-line labels sit
    # flush with the axis rather than centred across the gap.
    ax.set_xticklabels(col_labels, rotation=40, ha="right",
                       fontsize=8, rotation_mode="anchor",
                       multialignment="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Bold + rectangle for highlighted rows
    for tick, label in zip(ax.yaxis.get_major_ticks(), row_labels):
        if label in highlight_rows:
            tick.label1.set_fontweight("bold")

    for i, label in enumerate(row_labels):
        if label in highlight_rows:
            rect = Rectangle(
                (-0.5, i - 0.5), n_cols, 1.0,
                linewidth=1.5, edgecolor="black",
                facecolor="none", zorder=3, clip_on=False,
            )
            ax.add_patch(rect)

    ax.tick_params(axis="both", which="both", length=0, pad=3)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    # Clip slightly outside the axis so the border rectangles show fully
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    return im


# ==========================================================================
# 5. Console report
# ==========================================================================

def _report_split(model_labels, model_accs, attack_list, split):
    split_name = "Development" if split == "dev" else "Evaluation"
    print(f"\n{'='*64}")
    print(f"  {split_name} set")
    print(f"{'='*64}")
    print(f"  {'Model':<22s}  {'Avg':>6s}  Worst three attacks")
    print(f"  {'-'*22}  {'-'*6}  {'-'*34}")
    for label, acc in zip(model_labels, model_accs):
        flat_label = label.replace("\n", " ")
        split_acc = acc.get(split, {})
        vals = {a: split_acc[a] for a in attack_list if a in split_acc}
        if not vals:
            print(f"  {flat_label:<22s}  {'—':>6s}  (no data)")
            continue
        avg = np.nanmean(list(vals.values()))
        worst = sorted(vals, key=lambda a: vals[a])[:3]
        worst_str = ",  ".join(f"{a}={vals[a]:.1f}%" for a in worst)
        print(f"  {flat_label:<22s}  {avg:>5.2f}%  {worst_str}")


# ==========================================================================
# 6. Main
# ==========================================================================

def main():
    # -----------------------------------------------------------------------
    # Parse and compute
    # -----------------------------------------------------------------------
    print("Parsing input files...")
    model_labels = [m["label"] for m in MODELS]
    model_accs   = []
    for m in MODELS:
        label = m["label"].replace("\n", " ")
        path  = m["path"]
        print(f"  {label}: {path or '(skipped — path=None)'}")
        raw = parse_file(path)
        model_accs.append(compute_accuracy(raw))

    # -----------------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------------
    _report_split(model_labels, model_accs, DEV_ATTACKS,  "dev")
    _report_split(model_labels, model_accs, EVAL_ATTACKS, "eval")
    print(
        "\nNote: '—' cells in the Dev panel for Spooftral Enc are expected —\n"
        "  the classification-head EER file does not include a per-attack\n"
        "  Dev breakdown (only Evaluation set eval stats is present).\n"
        "  '—' cells for Spooftral + Margin mean the checkpoint file path\n"
        "  is not yet set (path=None in MODELS)."
    )

    # -----------------------------------------------------------------------
    # Build matrices
    # -----------------------------------------------------------------------
    mat_dev  = build_matrix(model_accs, DEV_ATTACKS,  "dev")
    mat_eval = build_matrix(model_accs, EVAL_ATTACKS, "eval")

    # Single shared normalization so one colorbar covers both panels.
    vmin_global = float(min(np.nanmin(mat_dev), np.nanmin(mat_eval)))
    vmax_global = float(max(np.nanmax(mat_dev), np.nanmax(mat_eval)))

    # Red → Yellow → Green colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ryg",
        ["#D7191C", "#FDAE61", "#A6D96A", "#1A9641"],
        N=256,
    )

    # Column labels
    col_labels = [m["label"] for m in MODELS]

    # -----------------------------------------------------------------------
    # Figure layout: dev panel | eval panel | single colorbar
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[len(DEV_ATTACKS), len(EVAL_ATTACKS), 0.5],
    )

    ax_dev  = fig.add_subplot(gs[0, 0])
    ax_eval = fig.add_subplot(gs[0, 1])
    cax     = fig.add_subplot(gs[0, 2])

    # -----------------------------------------------------------------------
    # Draw heatmaps (both use the same vmin/vmax)
    # -----------------------------------------------------------------------
    im_dev = plot_heatmap(
        ax_dev, mat_dev,
        row_labels=DEV_ATTACKS,
        col_labels=col_labels,
        title="(a) ASVspoof5 Development",
        highlight_rows=HIGHLIGHT_DEV,
        vmin=vmin_global, vmax=vmax_global,
        cmap=cmap,
        fig=fig,
    )

    im_eval = plot_heatmap(
        ax_eval, mat_eval,
        row_labels=EVAL_ATTACKS,
        col_labels=col_labels,
        title="(b) ASVspoof5 Evaluation",
        highlight_rows=HIGHLIGHT_EVAL,
        vmin=vmin_global, vmax=vmax_global,
        cmap=cmap,
        fig=fig,
    )

    # -----------------------------------------------------------------------
    # Single colorbar on the right of the eval panel
    # -----------------------------------------------------------------------
    cb = fig.colorbar(im_eval, cax=cax, orientation="vertical")
    cb.set_label("Detection Accuracy (%)", fontsize=7, labelpad=4)
    cb.ax.tick_params(labelsize=6.5)
    ticks = sorted({round(vmin_global, 1), 50.0, 75.0, 90.0, round(vmax_global, 1)})
    ticks = [t for t in ticks if vmin_global <= t <= vmax_global]
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_tick_params(which="both", length=2)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out = f"{OUTPUT_PREFIX}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", format="png")
    print(f"\nSaved: {out}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
