"""App. "Full single qubit tomography": Fig. pairclamp, Table modes and the quoted claims.

Stored data: ``results/simulation/pair_encoding/pair_encoding_results.json``, written by
the original analysis script of the paper (not published; experiments ``e1``-``e4``). Fig.
pairclamp is drawn from its ``e2_clamp`` block (noiseless Aer, 4096 shots per basis,
4x4 lattice, payload seeds 11-13). The ``rerun_*`` functions port the four
experiments to the :mod:`tessa` API so the JSON can be regenerated; they return the
blocks in the stored schema and write nothing.

Costs (4-core VM): ``rerun_e1`` about 1 s, ``rerun_e2`` about 10 s, ``rerun_e3``
about 15 s, ``rerun_e4`` about 8 s. All four reproduce every stored number and
reconstruction exactly (same seeds, bare cells ``halo=None``, FakeFez of
qiskit-ibm-runtime 0.52.1).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
from matplotlib.figure import Figure
from PIL import Image
from qiskit.quantum_info import Statevector

from tessa import (
    MODES,
    Lattice,
    aer_sampler,
    apply_dynamic_range,
    cell_circuit,
    decode,
    decode_from_cell_counts,
    encode,
    fake_sampler,
    graph_from_backend,
)
from tessa.core import get_mode
from tessa_paper._check import kind_of, with_kinds
from tessa_paper.paths import DATA, RESULTS

__all__ = [
    "KNOWN_DISCREPANCIES",
    "PAPER_VALUES",
    "RESULTS_FILE",
    "check_against_paper",
    "claims",
    "compare_rerun",
    "fig_pair_clamp",
    "load_results",
    "predicted_variance",
    "rerun_e1",
    "rerun_e2",
    "rerun_e3",
    "rerun_e4",
    "table_modes",
]

RESULTS_FILE = RESULTS / "simulation" / "pair_encoding" / "pair_encoding_results.json"
QCAT_COLOR = DATA / "images" / "qcat" / "qcat_color.png"
UNIT = (0.0, 1.0)
PI = np.pi
#: The paper's default polar clamp, in units of pi.
DEFAULT_CLAMP_OVER_PI = 0.2

#: Numbers and claims printed in App. E (Fig. 11, Table 5 and the text).
PAPER_VALUES: dict[str, object] = {
    "fig:pairclamp shots per basis": 4096,
    "fig:pairclamp lattice": "4x4",
    "fig:pairclamp payloads": 3,
    "default clamp c/pi": 0.2,
    # "At the default clamp the second field's error is about three times lower than on
    # the half circle" (App. E text).
    "full/half circle second-field error ratio": "about 3",
    "full circle second field below single encoding": True,
    "full circle first field improves slightly": True,
    "half circle, c_b = c: best second-field clamp c/pi": "about 0.2",
    "half circle, c_b = 0: second field keeps improving with c": True,
    "tab:modes": {
        "Z": ("C", "---", "---", "1"),
        "Z,X": ("2C", "[0,pi]", "theta", "1"),
        "Z,X,Y": ("2C", "[0,2pi)", "theta, phi*", "1.5"),
        "Z,X,Y (sign bit)": ("2C + C bits", "[0,pi] + sign", "theta, phi", "1.5"),
    },
}

#: No discrepancy between the stored data and the printed text is known here.
KNOWN_DISCREPANCIES: dict[str, str] = {}


def load_results(path: Path = RESULTS_FILE) -> dict:
    """The stored ``pair_encoding_results.json`` (keys e1_exact, e2_clamp, e3_colour, e4_sign_bit)."""
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Fig. pairclamp
# ---------------------------------------------------------------------------
_STYLES = {
    "pair_cb0": ("$Z,X$ readout, $c_b=0$", "C3", "s"),
    "pair_cbc": ("$Z,X$ readout, $c_b=c$", "C0", "o"),
    "pair_full": ("$Z,X,Y$ readout, $c_b=c$", "C2", "^"),
}


def fig_pair_clamp(save: Path | None = None, e2: dict | None = None) -> Figure:
    """Fig. pairclamp: sampled RMS errors vs clamp, with the delta-method predictions.

    Input: the ``e2_clamp`` block of ``pair_encoding_results.json`` (or ``e2``, e.g.
    from :func:`rerun_e2`). Returns the figure; saves it at 220 dpi if ``save`` is given
    (paper file name ``fig_pair_clamp.png``).
    """
    d = load_results()["e2_clamp"] if e2 is None else e2
    with plt.rc_context({"font.size": 9, "axes.titlesize": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
        for name, (label, colour, marker) in _STYLES.items():
            rows = d[name]
            cs = [r["clamp_over_pi"] for r in rows]
            for ax, key in zip(axes, ("a", "b"), strict=True):
                ax.plot(
                    cs,
                    [r[f"rms_{key}"] for r in rows],
                    marker,
                    color=colour,
                    ms=4,
                    label=label,
                )
                if name != "pair_cb0":
                    ax.plot(cs, [r[f"pred_{key}"] for r in rows], "-", color=colour, lw=1)
        axes[0].axhline(
            d["single"]["rms_a"], color="k", ls=":", lw=1, label="single encoding ($Z$)"
        )
        titles = ("first field ($\\theta$, from $Z$)", "second field ($\\varphi$)")
        for ax, title in zip(axes, titles, strict=True):
            ax.set_yscale("log")
            ax.set_xlabel("clamp $c/\\pi$")
            ax.set_title(title)
            ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel("RMS error (value units)")
        axes[1].set_ylim(top=0.6)
        axes[0].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=220)
    return fig


# ---------------------------------------------------------------------------
# Table modes and the text claims
# ---------------------------------------------------------------------------
def table_modes() -> pd.DataFrame:
    """Table modes, derived from the mode definitions of :data:`tessa.MODES`.

    No stored data: values per slot follow from the number of fields, shots per value
    = number of bases / number of continuous values, relative to the single encoding
    (C = capacity). The sign-bit row is the ``pair_full`` readout used for a half
    circle plus one bit per slot (App. sec:xyz). Returns one row per readout.
    """
    rows = []
    for mode in ("single", "pair", "pair_full"):
        m = MODES[mode]
        rows.append(
            {
                "bases": ",".join(b.upper() for b in m.bases),
                "values": "C" if m.n_fields == 1 else f"{m.n_fields}C",
                "phi": "---"
                if m.n_fields == 1
                else ("[0,2pi)" if m.full_circle else "[0,pi]"),
                "clamped": "---"
                if m.n_fields == 1
                else ("theta, phi*" if m.full_circle else "theta"),
                "shots_per_value": len(m.bases) / m.n_fields,
            }
        )
    rows.append(
        {
            "bases": "Z,X,Y (sign bit)",
            "values": "2C + C bits",
            "phi": "[0,pi] + sign",
            "clamped": "theta, phi",
            "shots_per_value": len(MODES["pair_full"].bases) / 2,
        }
    )
    return pd.DataFrame(rows).set_index("bases")


def claims(e2: dict | None = None) -> dict[str, float | bool]:
    """The numbers behind the appendix's statements, from the ``e2_clamp`` block.

    Returns RMS errors at the default clamp c = 0.2 pi, the ratio of the half-circle to
    the full-circle second-field error (for c_b = c and c_b = 0), the clamp minimising
    the half-circle (c_b = c) second-field error, whether the c_b = 0 error decreases
    monotonically with c, and the worst |sampled/predicted - 1| with and without the
    smallest clamp.
    """
    d = load_results()["e2_clamp"] if e2 is None else e2

    def at(name: str, c: float = DEFAULT_CLAMP_OVER_PI) -> dict:
        return next(r for r in d[name] if abs(r["clamp_over_pi"] - c) < 1e-9)

    full, cbc, cb0 = at("pair_full"), at("pair_cbc"), at("pair_cb0")
    rel = {
        name: [
            max(abs(r["rms_a"] / r["pred_a"] - 1), abs(r["rms_b"] / r["pred_b"] - 1))
            for r in d[name]
        ]
        for name in ("pair_cbc", "pair_full")
    }
    cb0_b = [r["rms_b"] for r in d["pair_cb0"]]
    return {
        "single rms_a": d["single"]["rms_a"],
        "pair_full rms_a @0.2pi": full["rms_a"],
        "pair_full rms_b @0.2pi": full["rms_b"],
        "pair_cbc rms_a @0.2pi": cbc["rms_a"],
        "pair_cbc rms_b @0.2pi": cbc["rms_b"],
        "pair_cb0 rms_b @0.2pi": cb0["rms_b"],
        "ratio half(c_b=c)/full second field": cbc["rms_b"] / full["rms_b"],
        "ratio half(c_b=0)/full second field": cb0["rms_b"] / full["rms_b"],
        "best clamp half circle c_b=c": min(d["pair_cbc"], key=lambda r: r["rms_b"])[
            "clamp_over_pi"
        ],
        "c_b=0 second field monotone in c": bool(np.all(np.diff(cb0_b) < 0)),
        "max |sampled/pred - 1|, all clamps": max(max(v) for v in rel.values()),
        "max |sampled/pred - 1|, c >= 0.1pi": max(max(v[1:]) for v in rel.values()),
    }


def check_against_paper() -> pd.DataFrame:
    """Every number and claim of the appendix that comes from the stored JSON.

    Returns a DataFrame [item, paper, reproduced, match, kind]. "About 3" is matched by
    a ratio in [2.5, 3.5] (the two ratios are 3.4 and 3.1); "about 0.2" by the
    minimising clamp being 0.2 pi. The number of random payloads per point (three)
    is not stored in the JSON and is therefore not compared here; :func:`rerun_e2`
    reproduces every stored number with three payloads (seeds 11-13).
    """
    data = load_results()
    d = data["e2_clamp"]
    c = claims(d)
    rows = [
        (
            "fig:pairclamp shots per basis",
            "4096",
            str(d["shots"]),
            d["shots"] == 4096,
            "number",
        ),
        ("fig:pairclamp lattice", "4x4", d["lattice"], d["lattice"] == "4x4", "label"),
    ]
    for key in (
        "ratio half(c_b=c)/full second field",
        "ratio half(c_b=0)/full second field",
    ):
        rows.append(
            (f"{key} @0.2pi", "about 3", f"{c[key]:.2f}", 2.5 <= c[key] <= 3.5, "claim")
        )
    rows += [
        (
            "full-circle second field < single encoding (rms)",
            "True",
            f"{c['pair_full rms_b @0.2pi']:.4f} < {c['single rms_a']:.4f}",
            c["pair_full rms_b @0.2pi"] < c["single rms_a"],
            "claim",
        ),
        (
            "full-circle first field improves slightly (rms)",
            "True",
            f"{c['pair_full rms_a @0.2pi']:.4f} < {c['pair_cbc rms_a @0.2pi']:.4f}",
            c["pair_full rms_a @0.2pi"] < c["pair_cbc rms_a @0.2pi"],
            "claim",
        ),
        (
            "half circle, c_b=c: best clamp c/pi",
            "about 0.2",
            f"{c['best clamp half circle c_b=c']:.2f}",
            abs(c["best clamp half circle c_b=c"] - 0.2) < 1e-9,
            "claim",
        ),
        (
            "half circle, c_b=0: keeps improving with c",
            "True",
            str(c["c_b=0 second field monotone in c"]),
            c["c_b=0 second field monotone in c"],
            "claim",
        ),
        (
            "sampled vs Eq. sigmapair: agree except smallest clamp (max rel. dev.)",
            "agree",
            f"{c['max |sampled/pred - 1|, c >= 0.1pi']:.2f} (c>=0.1pi), "
            f"{c['max |sampled/pred - 1|, all clamps']:.2f} (all)",
            c["max |sampled/pred - 1|, c >= 0.1pi"] < 0.2,
            "claim",
        ),
    ]
    modes = table_modes()
    for (bases, printed), (_, row) in zip(
        PAPER_VALUES["tab:modes"].items(), modes.iterrows(), strict=True
    ):
        got = f"{row['shots_per_value']:g}"
        rows.append(
            (
                f"tab:modes {bases} shots/value",
                printed[3],
                got,
                got == printed[3],
                kind_of(printed[3]),
            )
        )
    return with_kinds(
        pd.DataFrame(rows, columns=["item", "paper", "reproduced", "match", "kind"])
    )


# ---------------------------------------------------------------------------
# Re-running the experiments with the tessa API
# ---------------------------------------------------------------------------
def _encode_fields(graph, fields, mode: str, **kw):
    if mode == "single":
        return encode(graph, fields[0], mode="single", value_range=UNIT, **kw)
    return encode(
        graph, fields[0], fields[1], mode=mode, value_range=UNIT, value_range_b=UNIT, **kw
    )


def _exact_decode_per_group(circuit, enc):
    """Decode from exact cell probabilities (no shots, no noise)."""
    mode = get_mode(enc.mode)
    by_site = {}
    for g in enc.graph.groups:
        cell = cell_circuit(circuit, g)
        by_site[g.site] = {}
        for basis in mode.bases:
            qc = cell.copy()
            if basis == "x":
                qc.h(0)
            elif basis == "y":
                qc.sdg(0)
                qc.h(0)
            by_site[g.site][basis] = Statevector(qc).probabilities_dict()
    return decode_from_cell_counts(by_site, enc)


def rerun_e1() -> dict:
    """Experiment e1: exact whole-chip retrieval on the FakeFez graph, four modes.

    Payload ``default_rng(1)``; per-cell exact statevector probabilities. Returns the
    ``e1_exact`` block (max |error| per field; the stored ones are 1e-16 to 1e-13).
    """
    from tessa.backends.ibm import fake_backend

    graph = graph_from_backend(fake_backend("fez"))
    rng = np.random.default_rng(1)
    out: dict = {
        "device": "ibm_fez",
        "n_qubits": graph.n_qubits,
        "capacity": graph.capacity,
    }
    configs = [
        ("single", {}),
        ("pair", {}),
        ("pair", {"clamp_b": 0.1 * PI}),
        ("pair_full", {}),
    ]
    for mode, kw in configs:
        n_fields = get_mode(mode).n_fields
        fields = [rng.random(graph.capacity) for _ in range(n_fields)]
        circuit, enc = _encode_fields(graph, fields, mode, **kw)
        got = _exact_decode_per_group(circuit, enc)
        got = [got] if n_fields == 1 else list(got)
        errs = [float(np.abs(g - f).max()) for g, f in zip(got, fields, strict=True)]
        label = mode + ("" if not kw else f"_clampb{kw['clamp_b'] / PI:.1f}pi")
        out[label] = {
            "values": graph.capacity * n_fields,
            "clamps_over_pi": [c / PI for c in enc.clamps],
            "max_abs_error_per_field": errs,
        }
    return out


def predicted_variance(mode: str, x, y, deg, shots: int, c: float, cb: float):
    """Delta-method variance of each decoded value, in value units (Eqs. sigmapair/full).

    ``x``/``y`` are the two fields in [0, 1]; each address configuration of a
    degree-``deg`` cell receives ``shots / 2**deg`` shots per basis. Returns
    ``(var_a, var_b)`` (``var_b`` is ``None`` for ``mode="single"``).
    """
    n = shots / 2.0**deg
    theta = c + (PI - 2 * c) * x
    s, z = np.sin(theta), np.cos(theta)
    if mode == "single":
        return 1.0 / n / (PI - 2 * c) ** 2, None
    if mode == "pair":
        phi = cb + (PI - 2 * cb) * y
        cp, sp = np.cos(phi), np.sin(phi)
        var_t = 1.0 / n
        var_p = (1.0 + cp**2 * np.cos(2 * theta)) / (n * s**2 * sp**2)
        return var_t / (PI - 2 * c) ** 2, var_p / (PI - 2 * cb) ** 2
    phi = cb + (2 * PI - 2 * cb) * y
    x_, y_ = s * np.cos(phi), s * np.sin(phi)
    r2 = s**2
    var_t = (
        r2 * (1 - z**2)
        + z**2 * x_**2 * (1 - x_**2) / r2
        + z**2 * y_**2 * (1 - y_**2) / r2
    ) / n
    var_p = (1 - 2 * r2 * np.cos(phi) ** 2 * np.sin(phi) ** 2) / (r2 * n)
    return var_t / (PI - 2 * c) ** 2, var_p / (2 * PI - 2 * cb) ** 2


def rerun_e2(which: str = "all") -> dict:
    """Experiment e2 (Fig. pairclamp): clamp sweep under shot noise on a 4x4 lattice.

    Noiseless ``aer_sampler(lattice, seed=seed)``, 4096 shots per basis, payload seeds
    11, 12, 13, bare cells (``halo=None``, as in the paper). ``which`` is ``"all"``,
    ``"single"``, ``"pair_cb0"``, ``"pair_cbc"`` or ``"pair_full"``. Returns the
    ``e2_clamp`` block (only the requested entries); about 10 s for ``"all"``.
    """
    lattice = Lattice(4, 4)
    shots, seeds = 4096, (11, 12, 13)
    clamps = [round(k, 2) for k in np.arange(0.05, 0.46, 0.05)]
    deg = np.concatenate([[g.degree] * g.n_values for g in lattice.groups])
    data: dict = {"shots": shots, "lattice": "4x4", "clamps_over_pi": clamps}
    if which in ("all", "single"):
        errs = []
        for seed in seeds:
            x = np.random.default_rng(seed).random(lattice.capacity)
            circuit, enc = _encode_fields(lattice, [x], "single")
            got = decode(
                circuit, enc, aer_sampler(lattice, seed=seed), shots=shots, halo=None
            )
            errs.append(got - x)
        va, _ = predicted_variance("single", x, None, deg, shots, 0.0, 0.0)
        data["single"] = {
            "rms_a": float(np.sqrt(np.mean(np.square(errs)))),
            "pred_a": float(np.sqrt(np.mean(va))),
        }
    configs = {
        "pair_cb0": ("pair", lambda c: 0.0),
        "pair_cbc": ("pair", lambda c: c),
        "pair_full": ("pair_full", lambda c: c),
    }
    for name, (mode, cb_of) in configs.items():
        if which not in ("all", name):
            continue
        rows = []
        for c_pi in clamps:
            c, cb = c_pi * PI, cb_of(c_pi * PI)
            ea, eb, pa, pb = [], [], [], []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                x, y = rng.random(lattice.capacity), rng.random(lattice.capacity)
                circuit, enc = _encode_fields(lattice, [x, y], mode, clamp=c, clamp_b=cb)
                a, b = decode(
                    circuit, enc, aer_sampler(lattice, seed=seed), shots=shots, halo=None
                )
                ea.append(a - x)
                eb.append(b - y if mode == "pair" else (b - y + 0.5) % 1.0 - 0.5)
                va, vb = predicted_variance(mode, x, y, deg, shots, c, cb)
                pa.append(va)
                pb.append(vb)
            rows.append(
                {
                    "clamp_over_pi": c_pi,
                    "rms_a": float(np.sqrt(np.mean(np.square(ea)))),
                    "rms_b": float(np.sqrt(np.mean(np.square(eb)))),
                    "pred_a": float(np.sqrt(np.mean(pa))),
                    "pred_b": float(np.sqrt(np.mean(pb))),
                }
            )
        data[name] = rows
    return data


def _colour_qcat() -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(QCAT_COLOR).convert("RGB"), float) / 255
    hsv = rgb_to_hsv(rgb)
    return hsv[..., 2], hsv[..., 0]


def _hue_error(got, want) -> np.ndarray:
    return np.abs((np.asarray(got) - want + 0.5) % 1.0 - 0.5)


def rerun_e3(which: str = "all", halo: str | None = None) -> dict:
    """Experiment e3 (not shown in the paper): colour QCAT on FakeFez, per-qubit noise.

    ``data/images/qcat/qcat_color.png`` as HSV value (theta) and hue (phi);
    ``fake_sampler("fez", local=True, seed=0)``, 8192 shots per basis. ``which`` is
    ``"all"`` or one of ``noiseless_pair_full``, ``fez_two_single``, ``fez_pair``,
    ``fez_pair_full``. Returns the ``e3_colour`` block (only the requested runs).
    ``halo=None`` (bare cells) reproduces the stored block exactly.
    """
    value, hue = _colour_qcat()
    shots, seed = 8192, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip = fake_sampler("fez", local=True, seed=seed)
    graph = graph_from_backend(chip.backend())
    ideal = aer_sampler(graph, seed=seed)
    data: dict = {
        "shots_per_setting": shots,
        "device": "FakeFez, local",
        "runs": {},
        "recon": {},
    }
    plan = {
        "noiseless_pair_full": (ideal, "pair_full"),
        "fez_two_single": (chip, "two_single"),
        "fez_pair": (chip, "pair"),
        "fez_pair_full": (chip, "pair_full"),
    }
    for name, (sampler, mode) in plan.items():
        if which not in ("all", name):
            continue
        if mode == "two_single":
            cv, ev = _encode_fields(graph, [value], "single")
            ch, eh = _encode_fields(graph, [hue], "single")
            v = decode(cv, ev, sampler, shots=shots, halo=halo)
            h = decode(ch, eh, sampler, shots=shots, halo=halo)
            depth = cv.depth()
        else:
            circuit, enc = _encode_fields(graph, [value, hue], mode)
            v, h = decode(circuit, enc, sampler, shots=shots, halo=halo)
            depth = circuit.depth()
        data["runs"][name] = {
            "value_mae": float(np.abs(v - value).mean()),
            "hue_mae": float(_hue_error(h, hue).mean()),
            "value_rho": float(np.corrcoef(value.ravel(), np.asarray(v).ravel())[0, 1]),
            "hue_outliers_gt_0.25": float((_hue_error(h, hue) > 0.25).mean()),
            "logical_depth": depth,
            "settings": {"two_single": 2, "pair": 2, "pair_full": 3}[mode],
            "mean_std_value_mae": float(
                np.abs(apply_dynamic_range(v, value, method="mean_std") - value).mean()
            ),
        }
        data["recon"][name] = [np.asarray(v).tolist(), np.asarray(h).tolist()]
    return data


def rerun_e4(which: str = "all", halo: str | None = None) -> dict:
    """Experiment e4 (not shown in the paper): the sign of <Y> as an extra bit per slot.

    FakeFez graph, payload ``default_rng(4)``, clamps c = c_b = 0.2 pi, 8192 shots per
    basis; ``which`` is ``"all"``, ``"noiseless"`` or ``"fez"``
    (``fake_sampler("fez", local=True, seed=0)``). Returns the ``e4_sign_bit`` block.
    """
    shots, seed = 8192, 0
    c, cb = 0.2 * PI, 0.2 * PI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip = fake_sampler("fez", local=True, seed=seed)
    graph = graph_from_backend(chip.backend())
    rng = np.random.default_rng(4)
    a = rng.random(graph.capacity)
    b = rng.random(graph.capacity)
    bits = rng.integers(0, 2, graph.capacity)
    phi0 = cb + (PI - 2 * cb) * b
    phi = np.where(bits == 1, 2 * PI - phi0, phi0)
    circuit, enc = encode(
        graph,
        a,
        phi / (2 * PI),
        mode="pair_full",
        clamp=c,
        clamp_b=0.0,
        value_range=UNIT,
        value_range_b=UNIT,
    )
    out: dict = {
        "shots_per_setting": shots,
        "clamp_over_pi": c / PI,
        "clamp_b_over_pi": cb / PI,
        "bits": int(graph.capacity),
        "continuous_values": int(2 * graph.capacity),
    }
    for name, sampler in (("noiseless", aer_sampler(graph, seed=seed)), ("fez", chip)):
        if which not in ("all", name):
            continue
        a_got, x_got = decode(circuit, enc, sampler, shots=shots, halo=halo)
        phi_got = np.mod(np.asarray(x_got) * 2 * PI, 2 * PI)
        bit_got = (phi_got > PI).astype(int)
        phi0_got = np.where(bit_got == 1, 2 * PI - phi_got, phi_got)
        b_got = np.clip((phi0_got - cb) / (PI - 2 * cb), 0, 1)
        out[name] = {
            "bit_error_rate": float((bit_got != bits).mean()),
            "a_mae": float(np.abs(a_got - a).mean()),
            "b_mae_given_bit": float(np.abs(b_got - b)[bit_got == bits].mean()),
        }
    return out


def _flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}[{i}]", v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)


def compare_rerun(key: str, new: dict) -> pd.DataFrame:
    """Compare a re-run block with the stored one, number by number.

    ``key`` is the stored block (``e1_exact``, ``e2_clamp``, ``e3_colour``,
    ``e4_sign_bit``); the reconstructions of e3 are skipped. Returns a DataFrame
    [quantity, stored, rerun, abs_diff] over the numbers present in both.
    """
    stored, got = {}, {}
    _flatten("", {k: v for k, v in load_results()[key].items() if k != "recon"}, stored)
    _flatten("", {k: v for k, v in new.items() if k != "recon"}, got)
    rows = [
        {
            "quantity": k,
            "stored": stored[k],
            "rerun": got[k],
            "abs_diff": abs(stored[k] - got[k]),
        }
        for k in stored
        if k in got
    ]
    return pd.DataFrame(rows)
