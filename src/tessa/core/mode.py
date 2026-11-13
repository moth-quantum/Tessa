"""The three encoding modes, and the angle maps they imply.

A cell (one data qubit, one address configuration) is a qubit state
``cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>``. The mode says what it holds and
which measurement settings read it back:

``single``
    One value in ``theta``. Settings: ``Z``.
``pair``
    Two values, ``phi`` on the half-circle ``[0, pi]``. Settings: ``Z``, ``X``.
``pair_full``
    Two values, ``phi`` on the full circle. Settings: ``Z``, ``X``, ``Y``.

**Clamps.** One per field. ``clamps[0]`` holds ``theta`` inside
``[c, pi - c]``; ``clamps[1]`` holds ``phi`` inside ``[c, pi - c]`` on the
half-circle or ``[c, 2*pi - c]`` on the full circle. The pair modes require
``clamps[0] > 0``, since a cell at ``theta = 0``/``pi`` is ``|0>``/``|1>`` and
carries no phase. ``clamps[0]`` defaults to :data:`DEFAULT_CLAMP` for the pair
modes, ``clamps[1]`` to ``0`` on the half-circle and to ``clamps[0]`` on the
full one.

On the full circle, ``clamps[1] = 0`` identifies ``phi = 0`` and ``phi = 2*pi``
(``x01 = 0`` and ``x01 = 1`` decode to the same phase) and leaves no margin
around that wrap point, so shot noise on a value near either end can flip the
decoded value to near the other end. ``clamps[1] > 0`` opens a dead zone around
the wrap that real data never occupies, buying margin before noise can cross
it -- worth it whenever the field being encoded is not itself periodic, or
downstream code does not already treat ``0`` and ``1`` as equivalent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Default clamp for the pair modes.
DEFAULT_CLAMP = 0.2 * np.pi

__all__ = ["DEFAULT_CLAMP", "MODES", "Mode", "as_clamps", "get_mode"]


def as_clamps(clamps: float | Sequence[float]) -> tuple[float, ...]:
    """Accept one clamp for both fields, or one per field."""
    if np.isscalar(clamps):
        return (float(clamps), float(clamps))
    return tuple(float(c) for c in clamps)


@dataclass(frozen=True)
class Mode:
    name: str
    n_fields: int
    bases: tuple[str, ...]
    full_circle: bool

    @property
    def is_pair(self) -> bool:
        return self.n_fields == 2

    def angles(self, x01: np.ndarray, clamps) -> np.ndarray:
        """``[0, 1]`` -> net ``RY`` angle in ``[c, pi - c]``."""
        c = as_clamps(clamps)[0]
        return c + (np.pi - 2.0 * c) * np.asarray(x01, dtype=float)

    def values_from_angles(self, theta: np.ndarray, clamps) -> np.ndarray:
        c = as_clamps(clamps)[0]
        return (np.asarray(theta, dtype=float) - c) / (np.pi - 2.0 * c)

    def phases(self, x01: np.ndarray, clamps) -> np.ndarray:
        """``[0, 1]`` -> net ``RZ`` phase, on the half-circle or the full one."""
        c = as_clamps(clamps)[1]
        span = (2.0 if self.full_circle else 1.0) * np.pi - 2.0 * c
        return c + span * np.asarray(x01, dtype=float)

    def values_from_phases(self, phi: np.ndarray, clamps) -> np.ndarray:
        c = as_clamps(clamps)[1]
        phi = np.asarray(phi, dtype=float)
        if not self.full_circle:
            return (phi - c) / (np.pi - 2.0 * c)
        # Safe modulo: the arc the clamp excludes is where the branch cut sits.
        return (np.mod(phi, 2.0 * np.pi) - c) / (2.0 * np.pi - 2.0 * c)

    def resolve_clamps(
        self, clamp: float | None = None, clamp_b: float | None = None
    ) -> tuple[float, ...]:
        """``(clamp, clamp_b)`` filled in and checked, one entry per field.

        ``clamp`` defaults to ``0`` for one field and :data:`DEFAULT_CLAMP` for
        two; ``clamp_b`` defaults to ``0`` on the half-circle and to ``clamp``
        on the full one.
        """
        if clamp is None:
            clamp = DEFAULT_CLAMP if self.is_pair else 0.0
        clamp = float(clamp)
        if self.is_pair and clamp <= 0:
            raise ValueError(
                f"mode {self.name!r} needs clamp > 0: at theta=0/pi the state is "
                "|0>/|1> and has no phase to hold the second value"
            )
        if not 0 <= clamp < np.pi / 2:
            raise ValueError(f"clamp must lie in [0, pi/2), got {clamp}")

        if not self.is_pair:
            if clamp_b is not None:
                raise ValueError(f"mode {self.name!r} has one field; clamp_b is unused")
            return (clamp,)

        if clamp_b is None:
            clamp_b = clamp if self.full_circle else 0.0
        clamp_b = float(clamp_b)
        limit = np.pi if self.full_circle else np.pi / 2
        if not 0 <= clamp_b < limit:
            raise ValueError(f"clamp_b must lie in [0, {limit:.4f}), got {clamp_b}")
        return (clamp, clamp_b)


MODES = {
    m.name: m
    for m in (
        Mode("single", 1, ("z",), False),
        Mode("pair", 2, ("z", "x"), False),
        Mode("pair_full", 2, ("z", "x", "y"), True),
    )
}


def get_mode(mode: str | Mode) -> Mode:
    if isinstance(mode, Mode):
        return mode
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
    return MODES[mode]
