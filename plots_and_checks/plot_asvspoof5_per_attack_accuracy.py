"""
plot_asvspoof5_per_attack_accuracy.py

Reads per-attack Correct/Total statistics from a checkpoint EER text file
(ASVspoof5) and produces a two-panel publication-ready bar chart.

Usage:
    python plot_asvspoof5_per_attack_accuracy.py \
        --input <path/to/checkpoint-epochN_eer.txt>
"""

import argparse
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Attack-type lookup
# ---------------------------------------------------------------------------

_DEV_TYPE_MAP = {
    "A09": "TTS", "A10": "TTS", "A11": "TTS", "A12": "TTS", "A14": "TTS",
    "A13": "VC",  "A15": "VC",  "A16": "VC",
}

_EVAL_TYPE_MAP = {
    "A17": "TTS", "A19": "TTS", "A21": "TTS", "A22": "TTS",
    "A28": "TTS", "A29": "TTS",
    "A24": "VC",  "A25": "VC",  "A26": "VC",
    "A18": "AT",  "A20": "AT",  "A23": "AT",  "A27": "AT",
    "A30": "AT",  "A31": "AT",  "A32": "AT",
}

TYPE_COLORS = {
    "TTS": "#2166AC",   # blue
    "VC":  "#D6604D",   # red-orange
    "AT":  "#4DAC26",   # green
}


def get_attack_type(attack_id: str, split: str) -> str:
    """Return the attack-type string ('TTS', 'VC', 'AT') for an attack ID.

    Args:
        attack_id: e.g. 'A13'
        split: 'dev' or 'eval'

    Returns:
        Type string, or 'UNKNOWN' with a warning if not in the map.
    """
    mapping = _DEV_TYPE_MAP if split == "dev" else _EVAL_TYPE_MAP
    if attack_id not in mapping:
        print(f"  [WARNING] attack {attack_id!r} not found in {split} type map — "
              "labelled as 'UNKNOWN'.")
        return "UNKNOWN"
    return mapping[attack_id]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_eer_file(path: str) -> dict:
    """Parse the EER text file and return raw per-attack Correct/Total data.

    Returns:
        {
          'dev':  {'A09': (correct, total), ...},
          'eval': {'A17': (correct, total), ...},
        }

    Raises:
        FileNotFoundError if the file does not exist.
        ValueError if an expected section header is missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"EER file not found: {path}")

    with open(path, "r") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    results = {"dev": {}, "eval": {}}

    # Locate section markers
    def _find_line(keyword: str, start: int = 0) -> int:
        for i, ln in enumerate(lines[start:], start):
            if keyword in ln:
                return i
        return -1

    # ---- Dev section -------------------------------------------------------
    dev_header_idx = _find_line("Dev set eval stats")
    if dev_header_idx == -1:
        raise ValueError("Could not find 'Dev set eval stats' section in the file.")

    col_header_idx = _find_line("Attack  | Correct / Total", dev_header_idx)
    if col_header_idx == -1:
        raise ValueError("Could not find attack-column header after 'Dev set eval stats'.")

    results["dev"] = _parse_attack_block(lines, col_header_idx + 1)

    # ---- Eval section ------------------------------------------------------
    eval_header_idx = _find_line("Evaluation set eval stats")
    if eval_header_idx == -1:
        raise ValueError("Could not find 'Evaluation set eval stats' section in the file.")

    col_header_idx = _find_line("Attack  | Correct / Total", eval_header_idx)
    if col_header_idx == -1:
        raise ValueError("Could not find attack-column header after 'Evaluation set eval stats'.")

    results["eval"] = _parse_attack_block(lines, col_header_idx + 1)

    return results


def _parse_attack_block(lines: list, start: int) -> dict:
    """Read 'AXX | correct / total' lines starting at *start*.

    Stops when:
      - The line is empty.
      - The line starts with 'bonafide'.
      - The line starts with 'Codec'.

    Returns:
        {'AXX': (correct, total), ...}
    """
    data = {}
    for ln in lines[start:]:
        stripped = ln.strip()
        if not stripped:
            break
        if stripped.lower().startswith("bonafide"):
            break
        if stripped.lower().startswith("codec"):
            break
        if "|" not in stripped:
            break   # unexpected format — stop safely
        left, right = stripped.split("|", 1)
        attack_id = left.strip()
        # Expect "correct / total"
        parts = right.split("/")
        if len(parts) != 2:
            print(f"  [WARNING] Unexpected format on line: {ln!r} — skipping.")
            continue
        try:
            correct = int(parts[0].strip())
            total   = int(parts[1].strip())
        except ValueError:
            print(f"  [WARNING] Could not parse integers on line: {ln!r} — skipping.")
            continue
        data[attack_id] = (correct, total)
    return data


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def compute_accuracy(data: dict) -> dict:
    """Convert {attack: (correct, total)} → {attack: accuracy_pct}.

    Args:
        data: raw counts from parse_eer_file.

    Returns:
        {attack_id: float} accuracy in percent [0, 100].
    """
    acc = {}
    for attack_id, (correct, total) in data.items():
        if total == 0:
            print(f"  [WARNING] total=0 for attack {attack_id} — accuracy set to nan.")
            acc[attack_id] = float("nan")
        else:
            acc[attack_id] = correct / total * 100.0
    return acc


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_split(ax, accuracy: dict, title: str, split: str,
               highlight_attacks: list) -> None:
    """Draw a bar chart for one split onto *ax*.

    Args:
        ax: matplotlib Axes.
        accuracy: {attack_id: accuracy_pct} (ordered dict recommended).
        title: subplot title.
        split: 'dev' or 'eval', used to look up attack types.
        highlight_attacks: list of attack IDs to annotate with value labels.
    """
    attacks = list(accuracy.keys())
    values  = [accuracy[a] for a in attacks]
    types   = [get_attack_type(a, split) for a in attacks]
    colors  = [TYPE_COLORS.get(t, "#888888") for t in types]

    x = range(len(attacks))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5,
                  zorder=3)

    # Highlight specified bars with an accent edge and value label
    for i, attack in enumerate(attacks):
        if attack in highlight_attacks:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(1.8)
            # Value label above bar
            ax.text(
                i, values[i] + 0.8,
                f"{values[i]:.1f}%",
                ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color="black",
            )

    ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
    ax.set_ylabel("Detection Accuracy (%)", fontsize=7.5)
    ax.set_ylim(45, 101)
    ax.set_xticks(list(x))
    ax.set_xticklabels(attacks, rotation=45, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Spine cleanup
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def _report(split_name: str, accuracy: dict, split: str) -> None:
    print(f"\n{'='*52}")
    print(f"  {split_name} — per-attack accuracies")
    print(f"{'='*52}")
    for attack, acc in accuracy.items():
        atype = get_attack_type(attack, split)
        print(f"  {attack:>4s}  [{atype:>7s}]  {acc:6.2f}%")

    if accuracy:
        worst_k = sorted(accuracy, key=lambda a: accuracy[a])
        print(f"\n  Lowest-accuracy attacks:")
        for a in worst_k[:3]:
            print(f"    {a}: {accuracy[a]:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-attack detection accuracy from an ASVspoof5 EER file."
    )
    parser.add_argument(
        "--input", "-i",
        default=(
            "voxtral_clean-generation-asvspoof5-aug-bfloat16-2-constant-"
            "classification_as_generation-with-aug21/"
            "scores_text_outputs/checkpoint-epoch9_eer.txt"
        ),
        help="Path to the checkpoint EER text file.",
    )
    parser.add_argument(
        "--out-prefix", default="asvspoof5_per_attack_accuracy",
        help="Output filename prefix (no extension).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Parse
    # ------------------------------------------------------------------
    print(f"\nReading: {args.input}")
    try:
        raw = parse_eer_file(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Compute accuracies
    # ------------------------------------------------------------------
    dev_acc  = compute_accuracy(raw["dev"])
    eval_acc = compute_accuracy(raw["eval"])

    # ------------------------------------------------------------------
    # 3. Console report
    # ------------------------------------------------------------------
    _report("ASVspoof5 Dev",  dev_acc,  "dev")
    _report("ASVspoof5 Eval", eval_acc, "eval")

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    fig, (ax_dev, ax_eval) = plt.subplots(
        1, 2,
        figsize=(7.2, 2.8),
        constrained_layout=True,
    )

    plot_split(
        ax_dev,
        dev_acc,
        title="(a) ASVspoof5 Development",
        split="dev",
        highlight_attacks=["A13", "A15"],
    )
    plot_split(
        ax_eval,
        eval_acc,
        title="(b) ASVspoof5 Evaluation",
        split="eval",
        highlight_attacks=["A28"],
    )

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor=TYPE_COLORS["TTS"], label="TTS",
                       edgecolor="black", linewidth=0.5),
        mpatches.Patch(facecolor=TYPE_COLORS["VC"],  label="VC",
                       edgecolor="black", linewidth=0.5),
        mpatches.Patch(facecolor=TYPE_COLORS["AT"],  label="AT",
                       edgecolor="black", linewidth=0.5),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out_path = f"{args.out_prefix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", format="png")
    print(f"\nSaved: {out_path}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
