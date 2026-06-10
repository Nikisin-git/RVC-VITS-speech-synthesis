"""Read TensorBoard event files from RVC/TTS runs and render training curves.

Two figures are produced:
- training_curves.png: every loss component on shared axes with EMA-smoothed
  bright line + raw translucent line, plus a thin learning-rate trace on a
  twin Y axis. Vertical dashed lines mark checkpoint save steps.
- gan_balance.png: just the generator-total vs discriminator-total losses,
  to show GAN equilibrium.

Used both from the finalize hook of the trainers and from the live chart
widget in the progress dialog.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Curve:
    name: str
    steps: list[int]
    values: list[float]


# Scalar keys we look for in TensorBoard events. Order matters for the legend.
_RVC_SCALARS = {
    "loss/g/total": "loss_g_total",
    "loss/d/total": "loss_d_total",
    "loss/g/mel":   "loss_g_mel",
    "loss/g/kl":    "loss_g_kl",
    "loss/g/fm":    "loss_g_fm",
}
_RVC_LR_KEY = "learning_rate"
_RVC_GAN_PAIR = ("loss/g/total", "loss/d/total")

# Coqui-Trainer prefixes its scalar tags with a phase ('TRAIN/' or 'EVAL/').
# Plain tags below are matched as suffixes against the actual keys.
_TTS_SCALARS = {
    "loss_0":        "loss_0_gen",       # combined generator loss
    "loss_1":        "loss_1_disc",      # discriminator loss
    "loss_mel":      "loss_mel",
    "loss_kl":       "loss_kl",
    "loss_feat":     "loss_feat",
    "loss_duration": "loss_duration",
}
_TTS_GAN_PAIR = ("loss_0", "loss_1")


def _find_event_dirs(root: Path) -> list[Path]:
    """Find every directory that contains a tfevents file under root."""
    seen: set[Path] = set()
    # tfevents files are usually 'events.out.tfevents.*' (PyTorch/Coqui) but
    # we also match the older 'tfevents.*' just in case.
    for pattern in ("events.out.tfevents.*", "tfevents.*"):
        for p in root.rglob(pattern):
            seen.add(p.parent)
    return sorted(seen)


def read_scalars(event_dir: Path, tags: Iterable[str] | None = None) -> dict[str, Curve]:
    """Load scalar curves from TF-event files in `event_dir` and its children.
    Multiple event dirs (Coqui writes both root and 'eval/') are merged.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as e:
        raise RuntimeError(
            "Не установлен пакет tensorboard. Поставьте 'pip install tensorboard'."
        ) from e

    out: dict[str, Curve] = {}
    requested: set[str] | None = set(tags) if tags else None
    for d in _find_event_dirs(event_dir):
        acc = EventAccumulator(str(d), size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception:
            continue
        for tag in acc.Tags().get("scalars", []):
            if requested is not None and not any(tag.endswith(t) for t in requested):
                continue
            events = acc.Scalars(tag)
            steps = [e.step for e in events]
            values = [float(e.value) for e in events]
            if tag in out:
                # Merge later runs (Coqui appends on resume).
                out[tag].steps.extend(steps)
                out[tag].values.extend(values)
            else:
                out[tag] = Curve(name=tag, steps=steps, values=values)
    # Sort by step in case events arrived out of order.
    for c in out.values():
        order = sorted(range(len(c.steps)), key=lambda i: c.steps[i])
        c.steps = [c.steps[i] for i in order]
        c.values = [c.values[i] for i in order]
    return out


def ema_smooth(values: list[float], alpha: float = 0.7) -> list[float]:
    """Exponential moving average. alpha is the weight of the previous EMA."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * out[-1] + (1.0 - alpha) * v)
    return out


def _match_curve(scalars: dict[str, Curve], suffix: str) -> Curve | None:
    """Find a curve whose tag ends with the given short name (Coqui prefixes
    its tags with 'TRAIN/' / 'EVAL/', RVC uses tags as-is)."""
    if suffix in scalars:
        return scalars[suffix]
    for tag, curve in scalars.items():
        if tag.endswith("/" + suffix) or tag == suffix:
            return curve
    return None


def _match_eval_curve(scalars: dict[str, Curve], suffix: str) -> Curve | None:
    """Find the EVAL counterpart of a TRAIN scalar in Coqui logs."""
    candidates = [f"EVAL/avg_{suffix}", f"avg_{suffix}", f"EVAL/{suffix}"]
    for c in candidates:
        if c in scalars:
            return scalars[c]
        for tag, curve in scalars.items():
            if tag.endswith("/avg_" + suffix) or tag.endswith(c):
                return curve
    return None


# Distinct, print-friendly colours for the composite figure. Avoid neon and
# avoid pairs that look identical in greyscale.
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def plot_training_curves(
    scalars: dict[str, Curve],
    output_path: Path,
    framework: str,
    checkpoint_steps: list[int] | None = None,
    ema_alpha: float = 0.7,
    semilog: bool = True,
) -> Path | None:
    """Composite plot of every loss component. Returns the output path or
    None if there was nothing to plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = _RVC_SCALARS if framework == "rvc" else _TTS_SCALARS
    series: list[tuple[str, Curve, Curve | None]] = []
    for short, _label in keys.items():
        curve = _match_curve(scalars, short)
        if curve is None or not curve.values:
            continue
        eval_curve = _match_eval_curve(scalars, short) if framework == "tts" else None
        series.append((short, curve, eval_curve))

    if not series:
        return None

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.92, bottom=0.12)

    for (short, curve, eval_curve), colour in zip(series, _PALETTE):
        ax.plot(curve.steps, curve.values, color=colour, alpha=0.18, linewidth=0.9)
        smoothed = ema_smooth(curve.values, ema_alpha)
        ax.plot(curve.steps, smoothed, color=colour, linewidth=1.8, label=short)
        if eval_curve and eval_curve.values:
            ax.plot(
                eval_curve.steps, eval_curve.values,
                color=colour, linewidth=1.3, linestyle="--",
                label=f"{short} (eval)",
            )

    if framework == "rvc":
        lr_curve = _match_curve(scalars, _RVC_LR_KEY)
        if lr_curve and lr_curve.values:
            ax2 = ax.twinx()
            ax2.plot(lr_curve.steps, lr_curve.values, color="#7f7f7f",
                     linewidth=1.0, linestyle=":", label="learning_rate")
            ax2.set_ylabel("learning_rate", color="#555")
            ax2.tick_params(axis="y", labelcolor="#555")
            ax2.set_yscale("log")

    if checkpoint_steps:
        for step in checkpoint_steps:
            ax.axvline(step, color="#888", linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_xlabel("Шаг обучения")
    ax.set_ylabel("Значение функции потерь")
    ax.set_title("Кривые обучения " + ("RVC" if framework == "rvc" else "VITS / Coqui-TTS"))
    ax.grid(True, alpha=0.25)
    if semilog:
        ax.set_yscale("symlog", linthresh=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_gan_balance(
    scalars: dict[str, Curve],
    output_path: Path,
    framework: str,
    checkpoint_steps: list[int] | None = None,
    ema_alpha: float = 0.7,
) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen_key, disc_key = _RVC_GAN_PAIR if framework == "rvc" else _TTS_GAN_PAIR
    gen = _match_curve(scalars, gen_key)
    disc = _match_curve(scalars, disc_key)
    if not (gen and disc and gen.values and disc.values):
        return None

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.13)

    for curve, colour, label in (
        (gen, "#1f77b4", f"{gen_key} (генератор)"),
        (disc, "#d62728", f"{disc_key} (дискриминатор)"),
    ):
        ax.plot(curve.steps, curve.values, color=colour, alpha=0.18, linewidth=0.9)
        ax.plot(curve.steps, ema_smooth(curve.values, ema_alpha),
                color=colour, linewidth=2.2, label=label)

    if checkpoint_steps:
        for step in checkpoint_steps:
            ax.axvline(step, color="#888", linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_xlabel("Шаг обучения")
    ax.set_ylabel("Значение функции потерь")
    ax.set_title("Баланс GAN: генератор vs. дискриминатор")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_curves(
    event_root: Path,
    output_dir: Path,
    framework: str,
    checkpoint_steps: list[int] | None = None,
) -> dict[str, str]:
    """Read events under event_root, write training_curves.png and
    gan_balance.png into output_dir. Returns a dict with the written paths
    (or empty strings if nothing was generated)."""
    event_root = Path(event_root)
    output_dir = Path(output_dir)
    try:
        scalars = read_scalars(event_root)
    except Exception as e:
        print(f"WARN: cannot read TF events from {event_root}: {e}", flush=True)
        return {"training_curves": "", "gan_balance": ""}

    if not scalars:
        return {"training_curves": "", "gan_balance": ""}

    curves_path = plot_training_curves(
        scalars, output_dir / "training_curves.png",
        framework=framework, checkpoint_steps=checkpoint_steps,
    )
    balance_path = plot_gan_balance(
        scalars, output_dir / "gan_balance.png",
        framework=framework, checkpoint_steps=checkpoint_steps,
    )
    return {
        "training_curves": str(curves_path) if curves_path else "",
        "gan_balance": str(balance_path) if balance_path else "",
    }
