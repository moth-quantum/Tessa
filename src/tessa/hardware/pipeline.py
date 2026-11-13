"""Pixel images -> circuits to measure -> pixels back (the hardware-package pipeline).

This is the pipeline behind the pixel-art hardware runs of the paper: a *request*
(a dict, as stored in ``requests/<label>.json`` of a hardware package) is turned
into a :class:`Plan` by :func:`build_plan`, and the measured readouts are turned
back into images and metrics by :func:`finish`.

A request picks a *representation* and a *basis*:

``gray`` / ``palette``  (one number per pixel: luminance, or the pixel's index in
a palette made from the image's own colours, sorted dark to light)
    ``z``    the number in the amplitude (``RY``), read in Z.
    ``x``    the number in the phase (``RZ``, half circle, ``theta = pi/2``),
             read in X.
    ``zx``   two pixels per cell: one in the amplitude, one in the phase
             (half circle). Twice the pixels; Z and X readouts.
    ``zxy``  as ``zx``, but the phase uses the full circle: finer steps, and a
             third (Y) readout.
``palette`` is a set of symbols, so it is read back with ``tessa.symbols``:
    ``zx``       the ``"levels"`` scheme: the decoder snaps the amplitude to its
                 index first and reads the phase with that index's exact
                 ``sin(theta)``. The phase indices stay off ``0`` and ``pi``.
    ``sphere1``  the ``"constellation"`` scheme, one pixel per slot: ``k`` points
                 spread over the Bloch sphere, read in Z, X and Y.
    ``sphere2``  as ``sphere1``, two pixels per slot: ``k * k`` points.
``color``  (three numbers per pixel, in the colour space ``colorspace``: ``rgb``,
``hsv``, ``hsl`` or ``ycbcr``)
    The same bases as ``gray``. The channels are encoded as three stacked
    planes (channel 0 on top), so the mapping lays out each plane like an image.
``huevalue``  (value in the amplitude, hue in the phase; saturation is kept from
the input and never encoded)
    ``zx``   hue squeezed onto the half circle.
    ``zxy``  hue on the full circle.
``transition``  (two images: A in the amplitude, B in the phase)
    Read along axes tilted from Z (alpha = 0, image A) to X (alpha = pi/2,
    image B): the readout at angle ``alpha`` measures ``cos(alpha) Z + sin(alpha) X``
    on every data qubit. ``transition_values`` is
    ``gray`` (luminance) or ``palette``: one palette shared by both images, where
    a pixel transparent in only one image is the palette entry ``"transparent"``.

Transparent pixels (alpha < 128) are not encoded, and come back transparent.
Which slot each pixel takes is the request's ``mapping`` (see
:mod:`tessa.hardware.mapping`).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace

import numpy as np
from qiskit import QuantumCircuit

from ..codec import encode
from ..symbols import encode_symbols, symbols_from_counts
from ..transforms import DYNAMIC_RANGE_METHODS, apply_dynamic_range
from .mapping import assign

__all__ = [
    "COLOR_SPACES",
    "REPRESENTATIONS",
    "Plan",
    "build_plan",
    "finish",
    "ideal_frames",
    "luminance",
    "palette_of",
    "parse_image",
    "shared_palette",
    "shots_of",
]

REPRESENTATIONS = {
    "gray": ("z", "x", "zx", "zxy"),
    "palette": ("z", "x", "zx", "zxy", "sphere1", "sphere2"),
    "color": ("z", "x", "zx", "zxy"),
    "huevalue": ("zx", "zxy"),
    "transition": ("zx",),
}
UNIT = (0.0, 1.0)
DRR_METHODS = tuple(m for m in DYNAMIC_RANGE_METHODS if m != "none")
MAX_SHOTS = 100_000
MAX_ALPHAS = 15
#: Palette bases read back as symbols: basis -> (tessa.symbols scheme, per slot).
SYMBOL_BASES = {
    "zx": ("levels", 2),
    "sphere1": ("constellation", 1),
    "sphere2": ("constellation", 2),
}
#: Largest constellation: the decoder compares every slot with every point.
MAX_POINTS = 4096


@dataclass
class Plan:
    """What to measure, and what is needed to turn the results into pixels."""

    representation: str
    basis: str
    shape: tuple[int, int]
    mask: np.ndarray  # (h, w) bool: the pixels that are encoded
    # (circuit, Encoding) pairs. A symbol plan (extra["symbols"]) is decoded from
    # its counts instead of its values.
    readouts: list[tuple[QuantumCircuit, object]]
    slot: np.ndarray  # (layers * n,) capacity slot of each encoded pixel (and channel)
    part: np.ndarray  # same shape: 0 = amplitude (theta), 1 = phase (phi), 2 = both
    groups: list[dict]
    capacity: int
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------
def parse_image(image: dict) -> np.ndarray:
    """``{"width", "height", "rgba": base64}`` -> ``(h, w, 4)`` uint8."""
    w, h = int(image["width"]), int(image["height"])
    if not (0 < w <= 256 and 0 < h <= 256):
        raise ValueError(f"images must be at most 256 x 256 pixels, got {w} x {h}")
    raw = np.frombuffer(base64.b64decode(image["rgba"]), dtype=np.uint8)
    if raw.size != w * h * 4:
        raise ValueError(f"expected {w * h * 4} RGBA bytes, got {raw.size}")
    return raw.reshape(h, w, 4)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """``(..., 3)`` uint8 -> ``[0, 1]``."""
    return (rgb[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])) / 255.0


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """``(n, 3)`` in ``[0, 1]`` -> ``(n, 3)`` hue, saturation, value in ``[0, 1]``."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    v = rgb.max(axis=1)
    c = v - rgb.min(axis=1)
    s = np.where(v > 0, c / np.where(v > 0, v, 1), 0.0)
    safe = np.where(c > 0, c, 1)
    h = np.select(
        [c == 0, v == r, v == g],
        [0.0, ((g - b) / safe) % 6, (b - r) / safe + 2],
        (r - g) / safe + 4,
    )
    return np.stack([h / 6.0, s, v], axis=1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_hsv`."""
    h, s, v = (np.mod(hsv[:, 0], 1.0) * 6.0), hsv[:, 1], hsv[:, 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    out = np.zeros((h.size, 3))
    for k, (r, g, b) in enumerate(table):
        sel = i == k
        out[sel] = np.stack([r[sel], g[sel], b[sel]], axis=1)
    return out


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """``(n, 3)`` in ``[0, 1]`` -> hue, saturation, lightness in ``[0, 1]``."""
    h, s, v = rgb_to_hsv(rgb).T
    lum = v * (1 - s / 2)
    room = np.minimum(lum, 1 - lum)
    sat = np.where(room > 0, (v - lum) / np.where(room > 0, room, 1), 0.0)
    return np.stack([h, sat, lum], axis=1)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_hsl`."""
    h, sat, lum = hsl.T
    v = lum + sat * np.minimum(lum, 1 - lum)
    s = np.where(v > 0, 2 * (1 - lum / np.where(v > 0, v, 1)), 0.0)
    return hsv_to_rgb(np.stack([h, s, v], axis=1))


_YCC = np.array(
    [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]
)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """``(n, 3)`` in ``[0, 1]`` -> full-range BT.601 Y, Cb, Cr in ``[0, 1]``."""
    return rgb @ _YCC.T + np.array([0.0, 0.5, 0.5])


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_ycbcr`."""
    return (ycc - np.array([0.0, 0.5, 0.5])) @ np.linalg.inv(_YCC).T


#: name -> (channel names, from RGB, to RGB, index of a periodic hue channel or None)
COLOR_SPACES = {
    "rgb": (("R", "G", "B"), lambda x: x, lambda x: x, None),
    "hsv": (("H", "S", "V"), rgb_to_hsv, hsv_to_rgb, 0),
    "hsl": (("H", "S", "L"), rgb_to_hsl, hsl_to_rgb, 0),
    "ycbcr": (("Y", "Cb", "Cr"), rgb_to_ycbcr, ycbcr_to_rgb, None),
}


TRANSPARENT = "transparent"


def _hex_rgb(key: str) -> tuple[int, int, int]:
    return tuple(int(key.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))


def shared_palette(pixel_sets: list[np.ndarray], order=None):
    """One palette for several ``(n, 4)`` pixel arrays, transparency as an entry.

    Returns ``(keys, indices)``: ``keys`` are ``"#rrggbb"`` or ``"transparent"``,
    dark to light with transparent first, or in ``order`` (each key exactly once);
    ``indices`` holds each array's palette index per pixel.
    """
    key_sets = [
        [
            TRANSPARENT if a < 128 else f"#{r:02x}{g:02x}{b:02x}"
            for r, g, b, a in px.tolist()
        ]
        for px in pixel_sets
    ]
    present = sorted(set().union(*key_sets))
    if order:
        if len(set(order)) != len(order) or set(order) != set(present):
            raise ValueError("palette_order must list each of the images' colours once")
        keys = list(order)
    else:
        weight = [0.299, 0.587, 0.114]
        bright = {
            k: -1.0 if k == TRANSPARENT else float(np.dot(_hex_rgb(k), weight))
            for k in present
        }
        keys = sorted(present, key=bright.get)
    rank = {k: i for i, k in enumerate(keys)}
    return keys, [np.array([rank[k] for k in ks], dtype=int) for ks in key_sets]


def palette_of(rgb: np.ndarray, order=None) -> tuple[np.ndarray, np.ndarray]:
    """``(n, 3)`` uint8 -> ``(palette (k, 3), index (n,))``.

    The palette runs dark to light, or in ``order``: a list of ``"#rrggbb"``
    holding each of the image's colours exactly once. Index 0 is stored at the
    ``|0>`` end of the qubit's axis, the last index at the ``|1>`` end.
    """
    colours, index = np.unique(rgb, axis=0, return_inverse=True)
    if order:
        given = [
            tuple(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)) for h in order
        ]
        if len(set(given)) != len(given) or set(given) != set(
            map(tuple, colours.tolist())
        ):
            raise ValueError("palette_order must list each of the image's colours once")
        rank_of = {c: r for r, c in enumerate(given)}
        rank = np.array([rank_of[tuple(c)] for c in colours.tolist()])
        return np.array(given, dtype=colours.dtype), rank[index.ravel()]
    order = np.argsort(luminance(colours), kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return colours[order], rank[index.ravel()]


# ----------------------------------------------------------------------
# Encodings
# ----------------------------------------------------------------------
def _tilted(circuit: QuantumCircuit, encoding, alpha: float) -> QuantumCircuit:
    """``circuit`` turned so a Z measurement reads ``cos(a) Z + sin(a) X``."""
    out = circuit.copy()
    out.ry(-alpha, encoding.data_qubits)
    return out


def _angle_readout(encoding):
    """Decode a tilted circuit as one field: ``arccos(<Z>)`` per slot, in radians."""
    return replace(encoding, mode="single", clamps=(0.0,), ranges=((0.0, np.pi),))


def _readouts(graph, rep: str, basis: str, a: np.ndarray, b: np.ndarray | None, extra):
    """Circuits to measure for full-capacity fields ``a`` (amplitude), ``b`` (phase)."""
    if rep == "transition":
        circuit, enc = encode(
            graph, a, b, mode="pair", value_range=UNIT, value_range_b=UNIT
        )
        alphas = extra["alphas"]
        extra["clamps"] = enc.clamps
        return [(_tilted(circuit, enc, al), _angle_readout(enc)) for al in alphas]
    if basis == "z":
        return [encode(graph, a, value_range=UNIT)]
    if basis == "x":
        # theta = pi/2 everywhere (a constant 0.5 maps to the middle of any clamp),
        # so <X> = cos(phi) and the pixel is phi / pi.
        circuit, enc = encode(
            graph,
            np.full(a.size, 0.5),
            b,
            mode="pair",
            clamp_b=0.0,
            value_range=UNIT,
            value_range_b=UNIT,
        )
        return [(_tilted(circuit, enc, np.pi / 2), _angle_readout(enc))]
    mode = "pair" if basis == "zx" else "pair_full"
    # Hue is periodic, so on the full circle its 0 and 1 may meet (clamp_b = 0).
    clamp_b = 0.0 if rep == "huevalue" else None
    return [
        encode(
            graph,
            a,
            b,
            mode=mode,
            clamp_b=clamp_b,
            value_range=UNIT,
            value_range_b=UNIT,
        )
    ]


def _symbol_readouts(graph, basis: str, slot: np.ndarray, part: np.ndarray, extra):
    """A palette as symbols: the circuit, and ``extra["symbols"]`` to decode it.

    ``tessa.encode_symbols`` stores position ``j`` of every slot in the ``j``-th
    block of ``capacity`` symbols, so the mapping's ``(slot, part)`` of a pixel is
    index ``part * capacity + slot``. Unused slots hold index 0.
    """
    scheme, per_slot = SYMBOL_BASES[basis]
    k = len(extra["palette"])
    if scheme == "constellation" and k**per_slot > MAX_POINTS:
        most = int(MAX_POINTS ** (1 / per_slot))
        raise ValueError(
            f"a constellation of {k} colours needs {k**per_slot} points, more than "
            f"{MAX_POINTS}: use at most {most} colours with this basis"
        )
    symbols = np.zeros(per_slot * graph.capacity, dtype=int)
    symbols[part * graph.capacity + slot] = extra["index"]
    circuit, enc = encode_symbols(graph, symbols, k, scheme=scheme, per_slot=per_slot)
    extra["symbols"] = enc
    return [(circuit, enc.encoding)]


def build_plan(request: dict, graph) -> Plan:
    """The circuits and bookkeeping for one request on ``graph``.

    ``request`` holds ``images`` (``{"width", "height", "rgba": base64}``), and
    optionally ``representation``, ``basis``, ``mapping``, ``seed``, ``colorspace``,
    ``transition_values``, ``alphas``, ``palette_order`` and ``shots``, as listed in
    the module docstring.
    """
    rep = request.get("representation", "gray")
    if rep not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {rep!r}")
    basis = request.get("basis", REPRESENTATIONS[rep][0])
    if basis not in REPRESENTATIONS[rep]:
        raise ValueError(f"{rep!r} supports bases {REPRESENTATIONS[rep]}, not {basis!r}")

    images = [parse_image(im) for im in request.get("images", [])]
    if len(images) != (2 if rep == "transition" else 1):
        raise ValueError(f"{rep!r} needs {2 if rep == 'transition' else 1} image(s)")
    if any(im.shape != images[0].shape for im in images):
        raise ValueError("the two images must have the same size")
    mask = np.any([im[..., 3] >= 128 for im in images], axis=0)
    n = int(mask.sum())
    if n == 0:
        raise ValueError("the image has no opaque pixels")
    one_number = rep in ("gray", "palette", "color")
    layers = 3 if rep == "color" else 1
    per_slot = 2 if one_number and basis in ("zx", "zxy", "sphere2") else 1
    # Full colour just short of room: the last few pixels go grayscale (one number).
    n_gray = grayscale_needed(n, graph.capacity * per_slot) if layers == 3 else 0
    slots = -(-(layers * n - 2 * n_gray) // per_slot)
    if slots > graph.capacity:
        what = f"{n} opaque pixels" + (" x 3 channels" if layers == 3 else "")
        raise ValueError(f"{what} need {slots} slots, but this chip has {graph.capacity}")
    mapping = request.get("mapping", "lines")
    seed = int(request.get("seed", 0))
    planes = [mask]
    if layers == 3:
        colour = mask.copy()  # channels 1 and 2 skip the grayscale pixels
        colour.ravel()[np.flatnonzero(mask.ravel())[n - n_gray :]] = False
        planes += [colour, colour]
    slot, part = assign(mapping, _stacked(planes), graph, per_slot, seed)

    pixels = [im[mask] for im in images]  # (n, 4) each, raster order
    extra: dict = {"mapping": mapping, "planes": planes, "n_gray": n_gray}
    a = np.zeros(graph.capacity)  # unused slots stay at 0
    b = np.zeros(graph.capacity)
    if one_number:
        if rep == "gray":
            f = luminance(pixels[0])
        elif rep == "palette":
            palette, index = palette_of(pixels[0][:, :3], request.get("palette_order"))
            extra.update(palette=palette, index=index)
            f = index / max(len(palette) - 1, 1)
        else:
            space = request.get("colorspace", "rgb")
            if space not in COLOR_SPACES:
                raise ValueError(
                    f"unknown colour space {space!r}; use {sorted(COLOR_SPACES)}"
                )
            rgb = pixels[0][:, :3] / 255.0
            channels = np.clip(COLOR_SPACES[space][1](rgb), 0.0, 1.0)
            extra.update(colorspace=space, channels=channels, rgb=rgb)
            m = n - n_gray
            first = np.concatenate([channels[:m, 0], luminance(pixels[0][m:])])
            f = np.concatenate(
                [first, channels[:m, 1], channels[:m, 2]]
            )  # plane by plane
        extra["input"] = f
        if basis == "x":
            part = np.ones(part.size, dtype=int)
        a[slot[part == 0]] = f[part == 0]
        b[slot[part == 1]] = f[part == 1]
    elif rep == "huevalue":
        hsv = rgb_to_hsv(pixels[0][:, :3] / 255.0)
        extra["hsv"] = hsv
        a[slot], b[slot] = hsv[:, 2], hsv[:, 0]
        part = np.full(n, 2)
    else:
        kind = request.get("transition_values", "gray")
        if kind == "palette":
            keys, (ia, ib) = shared_palette(pixels, request.get("palette_order"))
            extra.update(transition_palette=keys, index=(ia, ib))
            fa, fb = (i / max(len(keys) - 1, 1) for i in (ia, ib))
        elif kind == "gray":
            fa, fb = (luminance(p) * (p[:, 3] >= 128) for p in pixels)
        else:
            raise ValueError(
                f"transition_values must be 'gray' or 'palette', not {kind!r}"
            )
        extra.update(inputs=(fa, fb), transition_values=kind)
        a[slot], b[slot] = fa, fb
        steps = int(np.clip(request.get("alphas", 9), 2, MAX_ALPHAS))
        extra["alphas"] = np.linspace(0.0, np.pi / 2, steps)
        part = np.full(n, 2)
    if rep == "palette" and basis in SYMBOL_BASES:
        readouts = _symbol_readouts(graph, basis, slot, part, extra)
    else:
        readouts = _readouts(graph, rep, basis, a, b, extra)

    groups = [
        {
            "data": g.data_qubit,
            "addresses": list(g.address_qubits),
            "n_values": g.n_values,
        }
        for g in graph.groups
    ]
    return Plan(
        rep, basis, mask.shape, mask, readouts, slot, part, groups, graph.capacity, extra
    )


def grayscale_needed(n: int, room: int) -> int:
    """How many of ``n`` full-colour pixels to store as grayscale to fit in ``room``
    numbers: each one saves two. ``0`` if all fit, or if more than 1% (at least 2)
    would have to go, which is no longer an edge case."""
    k = -(-(3 * n - room) // 2)
    return k if 0 < k <= max(2, n // 100) else 0


def _stacked(planes: list[np.ndarray]) -> np.ndarray:
    """The planes one above the other, each padded with transparent rows to a
    multiple of 8, so a tile never spans two channels."""
    if len(planes) == 1:
        return planes[0]
    pad = np.zeros((-planes[0].shape[0] % 8, planes[0].shape[1]), dtype=bool)
    return np.vstack([x for plane in planes for x in (plane, pad)])


def shots_of(request: dict) -> int:
    """Shots per circuit of ``request`` (default 8192), clipped to ``[100, MAX_SHOTS]``."""
    return int(np.clip(int(request.get("shots", 8192)), 100, MAX_SHOTS))


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def _rgba(plan: Plan, rgb01: np.ndarray, opaque: np.ndarray | None = None) -> str:
    """Encoded pixels' ``(n, 3)`` colours in ``[0, 1]`` -> base64 RGBA of the sprite.

    ``opaque`` (``(n,)`` bool) can make some encoded pixels transparent.
    """
    out = np.zeros((*plan.shape, 4), np.uint8)
    out[plan.mask, :3] = np.clip(np.rint(rgb01 * 255), 0, 255).astype(np.uint8)
    out[plan.mask, 3] = 255 if opaque is None else np.where(opaque, 255, 0)
    return base64.b64encode(out.tobytes()).decode()


def _gray(v: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(v, 0, 1)[:, None], 3, axis=1)


def _transition_frames(plan: Plan, decoded: list) -> list[np.ndarray]:
    """Every tilted readout made into an image, by inverting ``cos(alpha) Z + sin(alpha) X``."""
    clamp_a, clamp_b = plan.extra["clamps"]
    E = [np.cos(np.asarray(angle, float)) for angle in decoded]
    cos_theta = E[0]
    sin_theta = np.sin(np.arccos(np.clip(cos_theta, -1, 1)))
    frames = []
    for alpha, e_alpha in zip(plan.extra["alphas"], E, strict=True):
        ca, sa = np.cos(alpha), np.sin(alpha)
        w = ca / (ca + sa)
        clamp = np.arccos(w * np.cos(clamp_a) + (1 - w) * np.cos(clamp_b))
        e = (ca * cos_theta + (e_alpha - ca * cos_theta) / sin_theta) / (ca + sa)
        frames.append((np.arccos(np.clip(e, -1, 1)) - clamp) / (np.pi - 2 * clamp))
    return frames


def _gray_note(k: int) -> str:
    many = k > 1
    return (
        f"To make full colour fit, the last {k} opaque pixel{'s' if many else ''} "
        f"(bottom right) {'are' if many else 'is'} stored in grayscale."
    )


def _fields(plan: Plan, decoded: list):
    """The decoded numbers per field, the input's own numbers for the same fields,
    and the fields dynamic range recovery must leave alone (periodic hue)."""
    n = int(plan.mask.sum())
    rep = plan.representation
    if rep in ("gray", "palette", "color"):
        if plan.basis == "z":
            v = np.asarray(decoded[0], float)[plan.slot]
        elif plan.basis == "x":
            v = np.asarray(decoded[0], float)[plan.slot] / np.pi  # clamp_b = 0
        else:
            amp, phase = decoded[0]
            v = np.where(plan.part == 0, amp[plan.slot], phase[plan.slot])
        f = plan.extra["input"]
        if rep != "color":
            return {"v": v}, {"v": f}, set()
        m = n - plan.extra["n_gray"]
        cut = [slice(0, n), slice(n, n + m), slice(n + m, None)]
        hue = COLOR_SPACES[plan.extra["colorspace"]][3]
        fixed = set() if hue is None else {f"c{hue}"}
        return (
            {f"c{c}": v[s] for c, s in enumerate(cut)},
            {f"c{c}": f[s] for c, s in enumerate(cut)},
            fixed,
        )
    if rep == "huevalue":
        value, hue = (np.asarray(x)[plan.slot] for x in decoded[0])
        hsv = plan.extra["hsv"]
        return (
            {"value": value, "hue": hue},
            {"value": hsv[:, 2], "hue": hsv[:, 0]},
            {"hue"},
        )
    frames = [f[plan.slot] for f in _transition_frames(plan, decoded)]
    ideal = ideal_frames(plan)
    return (
        {f"frame{k}": x for k, x in enumerate(frames)},
        {f"frame{k}": x for k, x in enumerate(ideal)},
        set(),
    )


def ideal_frames(plan: Plan) -> list[np.ndarray]:
    """What each tilted readout gives without noise: the reference for a transition."""
    clamp_a, clamp_b = plan.extra["clamps"]
    a, b = plan.extra["inputs"]
    cos_theta = np.cos(clamp_a + (np.pi - 2 * clamp_a) * a)
    cos_phi = np.cos(clamp_b + (np.pi - 2 * clamp_b) * b)
    frames = []
    for alpha in plan.extra["alphas"]:
        w = np.cos(alpha) / (np.cos(alpha) + np.sin(alpha))
        clamp = np.arccos(w * np.cos(clamp_a) + (1 - w) * np.cos(clamp_b))
        e = w * cos_theta + (1 - w) * cos_phi
        frames.append((np.arccos(np.clip(e, -1, 1)) - clamp) / (np.pi - 2 * clamp))
    return frames


def _render(plan: Plan, fields: dict) -> tuple[dict, dict, float]:
    """Fields -> (images, metrics, score). The score is the mean error the best
    dynamic range recovery is chosen by: lower is better."""
    n = int(plan.mask.sum())
    rep = plan.representation
    if rep == "gray":
        v = fields["v"]
        err = float(np.abs(v - plan.extra["input"]).mean() * 255)
        return {"image": _rgba(plan, _gray(v))}, {"mean_abs_error_255": err}, err
    if rep == "palette":
        k = len(plan.extra["palette"])
        index = np.clip(np.rint(fields["v"] * max(k - 1, 1)), 0, k - 1).astype(int)
        return _render_palette(plan, index)
    if rep == "color":
        names, _, to_rgb, hue = COLOR_SPACES[plan.extra["colorspace"]]
        m = n - plan.extra["n_gray"]
        got = np.stack([fields["c0"][:m], fields["c1"], fields["c2"]], axis=1)
        want = plan.extra["channels"][:m]
        rgb = np.empty((n, 3))
        rgb[:m] = to_rgb(np.clip(got, 0, 1))
        rgb[m:] = _gray(fields["c0"][m:])  # the grayscale pixels
        rgb = np.clip(rgb, 0, 1)
        err = float(np.abs(rgb - plan.extra["rgb"]).mean() * 255)
        metrics = {"mean_abs_error_rgb_255": err}
        for c, name in enumerate(names):
            if c == hue:
                d = np.abs((got[:, c] - want[:, c] + 0.5) % 1.0 - 0.5)
                metrics[f"mean_abs_error_{name}_degrees"] = float(d.mean() * 360)
            else:
                d = np.abs(got[:, c] - want[:, c])
                metrics[f"mean_abs_error_{name}_255"] = float(d.mean() * 255)
        return {"image": _rgba(plan, rgb)}, metrics, err
    if rep == "huevalue":
        value, hue = fields["value"], fields["hue"]
        hsv = plan.extra["hsv"]
        rgb = hsv_to_rgb(
            np.stack([np.mod(hue, 1.0), hsv[:, 1], np.clip(value, 0, 1)], axis=1)
        )
        err = float(np.abs(rgb - hsv_to_rgb(hsv)).mean() * 255)
        dh = np.abs((hue - hsv[:, 0] + 0.5) % 1.0 - 0.5)
        metrics = {
            "mean_abs_error_rgb_255": err,
            "mean_abs_error_value_255": float(np.abs(value - hsv[:, 2]).mean() * 255),
            "mean_abs_error_hue_degrees": float(dh.mean() * 360),
        }
        return {"image": _rgba(plan, rgb)}, metrics, err
    frames = [fields[f"frame{k}"] for k in range(len(plan.extra["alphas"]))]
    ideal = ideal_frames(plan)
    if plan.extra["transition_values"] == "palette":
        return _render_palette_frames(plan, frames, ideal)
    a, b = plan.extra["inputs"]
    err = float(
        np.mean([np.abs(x - y).mean() for x, y in zip(frames, ideal, strict=True)]) * 255
    )
    metrics = {
        "mean_abs_error_A_at_0_255": float(np.abs(frames[0] - a).mean() * 255),
        "mean_abs_error_B_at_90_255": float(np.abs(frames[-1] - b).mean() * 255),
        "mean_abs_error_all_angles_255": err,
    }
    return {"frames": [_rgba(plan, _gray(x)) for x in frames]}, metrics, err


def _render_palette(plan: Plan, index: np.ndarray) -> tuple[dict, dict, float]:
    """Palette indices per encoded pixel -> (image, metrics, score).

    The score is the mean index error, as a wrong pixel on a levels axis usually
    lands on a nearby index. A constellation's nearby points pair unrelated
    indices, so there the score is the fraction of wrong pixels.
    """
    palette, want = plan.extra["palette"], plan.extra["index"]
    err = float(np.abs(index - want).mean())
    exact = float((index == want).mean())
    metrics = {
        "palette_size": len(palette),
        "pixels_exact": exact,
        "mean_index_error": err,
    }
    symbols = plan.extra.get("symbols")
    score = (
        1.0 - exact if symbols is not None and symbols.scheme == "constellation" else err
    )
    return {"image": _rgba(plan, palette[index] / 255.0)}, metrics, score


def _symbol_index(plan: Plan, counts: dict, method: str) -> np.ndarray:
    """The palette index of every encoded pixel, read from counts with one DRR
    method (applied per field for levels, per Bloch axis for a constellation)."""
    symbols = symbols_from_counts(counts, plan.extra["symbols"], drr=method)
    return symbols[plan.part * plan.capacity + plan.slot]


def _render_palette_frames(plan: Plan, frames, ideal) -> tuple[dict, dict, float]:
    """Transition frames snapped to the shared palette, transparency included."""
    keys = plan.extra["transition_palette"]
    k = len(keys)
    rgb = np.array([(0, 0, 0) if c == TRANSPARENT else _hex_rgb(c) for c in keys]) / 255.0
    solid = np.array([c != TRANSPARENT for c in keys])

    def snap(x):
        return np.clip(np.rint(x * max(k - 1, 1)), 0, k - 1).astype(int)

    got = [snap(x) for x in frames]
    want = [snap(x) for x in ideal]
    ia, ib = plan.extra["index"]
    err = float(np.mean([np.abs(g - w).mean() for g, w in zip(got, want, strict=True)]))
    metrics = {
        "palette_size": k,
        "pixels_exact_A_at_0": float((got[0] == ia).mean()),
        "pixels_exact_B_at_90": float((got[-1] == ib).mean()),
        "mean_index_error_all_angles": err,
    }
    images = {"frames": [_rgba(plan, rgb[g], solid[g]) for g in got]}
    return images, metrics, err


def _recover(recon: np.ndarray, reference: np.ndarray, method: str) -> np.ndarray:
    """One dynamic range recovery method; raw values wherever it has no answer."""
    if recon.size == 0:
        return recon
    out = apply_dynamic_range(recon, reference, method)
    return np.clip(np.where(np.isfinite(out), out, recon), 0.0, 1.0)


def finish(plan: Plan, decoded: list) -> dict:
    """Decoded readouts -> the result dict: the raw read-back, and the
    dynamic range recovery (DRR) method that brings it closest to the input."""
    n = int(plan.mask.sum())
    bounds = np.cumsum([g["n_values"] for g in plan.groups])
    group = np.searchsorted(bounds, plan.slot, side="right")
    start = np.concatenate([[0], bounds])[group]
    planes = plan.extra["planes"]  # one map per colour channel, channel 0 first
    pixel_group = np.full((len(planes), *plan.shape), -1)
    pixel_address = np.full((len(planes), *plan.shape), -1)
    pixel_part = np.full((len(planes), *plan.shape), -1)
    offset = 0
    for c, plane in enumerate(planes):
        part_of = slice(offset, offset + int(plane.sum()))
        offset = part_of.stop
        pixel_group[c][plane] = group[part_of]
        pixel_address[c][plane] = (plan.slot - start)[part_of]
        pixel_part[c][plane] = plan.part[part_of]

    out = {
        "representation": plan.representation,
        "basis": plan.basis,
        "width": plan.shape[1],
        "height": plan.shape[0],
        "n_pixels": n,
        "capacity": plan.capacity,
        "slots_used": int(np.unique(plan.slot).size),
        "mapping": plan.extra["mapping"],
        "groups": plan.groups,
        "pixel_group": pixel_group.ravel().tolist(),
        "pixel_address": pixel_address.ravel().tolist(),
        "pixel_part": pixel_part.ravel().tolist(),
    }
    if plan.representation == "palette":
        out["palette"] = plan.extra["palette"].tolist()
    if plan.representation == "transition":
        out["alphas_degrees"] = np.degrees(plan.extra["alphas"]).tolist()
        out["transition_values"] = plan.extra["transition_values"]
        if plan.extra["transition_values"] == "palette":
            out["transition_palette"] = plan.extra["transition_palette"]
    if plan.representation == "color":
        out["channels"] = list(COLOR_SPACES[plan.extra["colorspace"]][0])
        out["colorspace"] = plan.extra["colorspace"]
        out["grayscale_pixels"] = plan.extra["n_gray"]
        if plan.extra["n_gray"]:
            out["notes"] = [_gray_note(plan.extra["n_gray"])]

    if "symbols" in plan.extra:
        out["symbol_scheme"] = plan.extra["symbols"].scheme

        def rendered_with(method):
            return _render_palette(plan, _symbol_index(plan, decoded[0], method))

    else:
        fields, refs, fixed = _fields(plan, decoded)

        def rendered_with(method):
            if method == "none":
                return _render(plan, fields)
            recovered = {
                k: v if k in fixed else _recover(v, refs[k], method)
                for k, v in fields.items()
            }
            return _render(plan, recovered)

    images, metrics, raw_score = rendered_with("none")
    out.update(images, metrics=metrics)

    scores, best = {"none": raw_score}, None
    for method in DRR_METHODS:
        rendered = rendered_with(method)
        scores[method] = rendered[2]
        if best is None or rendered[2] < best[1][2]:
            best = (method, rendered)
    method, (images, metrics, _) = best
    out["drr"] = {"method": method, "metrics": metrics, "scores": scores, **images}
    return out
