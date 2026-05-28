from __future__ import annotations

from pathlib import Path


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyArrowPatch
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for figure generation. Install with: pip install .[figures]"
        ) from exc

    return plt, Rectangle, FancyArrowPatch


def _box(ax, Rectangle, x, y, w, h, label, fontsize=10, linewidth=1.4):
    rect = Rectangle((x, y), w, h, fill=False, linewidth=linewidth)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, FancyArrowPatch, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14, linewidth=1.2))


def figure_visible_output_vs_representation_path(path: Path) -> None:
    plt, Rectangle, FancyArrowPatch = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.4)

    ax.text(
        6.25,
        5.05,
        "Prompt and Output Are Not Enough When the Claim Depends on Representation",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    _box(ax, Rectangle, 0.6, 3.55, 2.0, 0.75, "Visible prompt", fontsize=10)
    _box(ax, Rectangle, 5.25, 3.55, 2.0, 0.75, "Model response", fontsize=10)
    _box(ax, Rectangle, 9.65, 3.55, 2.2, 0.75, "Incomplete\nreproducibility", fontsize=10)

    _arrow(ax, FancyArrowPatch, 2.6, 3.93, 5.25, 3.93)
    _arrow(ax, FancyArrowPatch, 7.25, 3.93, 9.65, 3.93)

    ax.text(0.6, 4.55, "Surface artifact", ha="left", va="center", fontsize=10, fontweight="bold")

    fields = [
        ("Input\nidentity", 0.6),
        ("Normalization\nprofile", 2.35),
        ("Tokenizer\ncontract", 4.1),
        ("Context\npromotion", 5.85),
        ("Runtime /\ndecoding", 7.6),
        ("Evidence\nrecord", 9.35),
    ]

    for i, (label, x) in enumerate(fields):
        _box(ax, Rectangle, x, 1.5, 1.35, 0.8, label, fontsize=9)
        if i > 0:
            _arrow(ax, FancyArrowPatch, fields[i - 1][1] + 1.35, 1.9, x, 1.9)

    _box(
        ax,
        Rectangle,
        1.4,
        0.35,
        9.7,
        0.65,
        "A publishable engineering artifact contains the representation path, not only the visible text.",
        fontsize=10,
    )

    ax.text(0.6, 2.55, "Engineering artifact", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.text(6.25, 0.05, "Credit: Hazem Ali", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def figure_minimal_evidence_record(path: Path) -> None:
    plt, Rectangle, FancyArrowPatch = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.set_axis_off()
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.5)

    ax.text(5.75, 6.1, "Minimal Representation Evidence Record", ha="center", va="center", fontsize=13, fontweight="bold")
    _box(ax, Rectangle, 4.35, 4.8, 2.8, 0.85, "Representation\nevidence record", fontsize=11, linewidth=1.6)

    items = [
        ("Input hash", 0.75, 3.65),
        ("Unicode summary", 3.0, 3.65),
        ("Normalization policy", 5.25, 3.65),
        ("Tokenizer identity", 7.5, 3.65),
        ("Token count /\ntruncation state", 1.9, 2.3),
        ("Retrieval decisions", 4.15, 2.3),
        ("Context hash", 6.4, 2.3),
        ("Decoding config", 8.65, 2.3),
    ]

    for label, x, y in items:
        _box(ax, Rectangle, x, y, 1.65, 0.75, label, fontsize=8.8)
        _arrow(ax, FancyArrowPatch, 5.75, 4.8, x + 0.825, y + 0.75)

    _box(
        ax,
        Rectangle,
        1.35,
        0.75,
        8.8,
        0.85,
        "Goal: reproduce the representation path without requiring permanent storage of raw sensitive content.",
        fontsize=10,
    )

    ax.text(5.75, 0.18, "Credit: Hazem Ali", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def generate_figures(out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    figure_visible_output_vs_representation_path(out / "figure-01-visible-output-vs-representation-path")
    figure_minimal_evidence_record(out / "figure-02-minimal-representation-evidence-record")
