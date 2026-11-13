# simulation/entanglement/ (App. "Entanglement of the encoded state")

Reference outputs of `tessa_paper.entanglement.compute_all()` (written 2026-11-12, tessa 1.0.0,
qiskit 2.6.2; about 30 s on four cores). Everything is an exact, noiseless statevector
computation on at most 20 qubits; no sampling. The original appendix script (not published)
printed the numbers and wrote only the PNGs; the port gives bit-identical statevectors and arrays
(checked against the original script run from a copy on the same aarch64 machine: maximum
difference 0 for every array) and, with matplotlib 3.10.9, byte-identical
`fig_entanglement_map.png` and `fig_area_law.png`. Recomputed on Linux x86_64, some negativities
and entropies differ in the last bits (up to 3.4e-15 in `pairs.npz`, 1.3e-14 in `area_law.npz`; `entanglement_map.npz` is identical),
from the eigenvalue routines of the linear-algebra library; no printed number changes.

| file | content |
|---|---|
| `entanglement_map.npz` | Fig. entmap. `image` (21, 21) gray values of `data/images/qcat/qcat.png` / 255; `K_h`, `K_v` (21, 21): coherence K = cos(pi (v_i - v_j) / 2) of the address qubit of the degree-one super-pixel storing the horizontal / vertical neighbour pair whose first pixel is (r, c), 1 where there is no neighbour; `S` (21, 21): entropy in bits, max(h((1+K_h)/2), h((1+K_v)/2)). |
| `area_law.npz` | Fig. arealaw. `rows` = [2, 3, 4, 5], `cols` = [2, 3, 4, 5], `seeds` = 0..11; `S_rows` (4, 12): entropy (bits) across the cut after column 1 of an R x 4 lattice, payload `default_rng(seed).random(capacity)`; `S_cols` (4, 12): entropy of the first column of a 4 x C lattice. |
| `pairs.npz` | Table pairs. 3 x 4 lattice, qubit q at row q // 4, column q % 4, data qubit where row + column is even (`is_data`, 12); `pairs` (66, 2) all qubit pairs (u < w); `distance` (66) Manhattan distance; `seeds` = 100..139; `negativity`, `mutual_information` (40, 66) in bits, per payload and pair. |
| `summary.json` | `table_pairs` (rows of Table pairs, unrounded), `area_law` (means and standard deviations), `identity_check_max_abs_diff` (S(rho_a) vs h((1+K_a)/2), 3 x 4 lattice, seed 7: 6.2e-15), `address_pair_example_negativity` (0.1830, the entangled address pair of the text). |

Reference values (also printed by the original script; they match the paper): Table pairs DA d=1
N 0.0309 / 0.3336, I 0.1889 / 1.3292; AA d=2 N < 1e-4 / 0.0091, I 0.0519 / 0.3956; DD d=2 N 0 / 0,
I 0.0183 / 0.2903; DA d=3 I 0.0024 / 0.0703; AA d=4 I 0.0006 / 0.0099; DD d=4 and DA d=5 all 0.
Area law: S = 0.846, 1.301, 1.738, 2.155 (R = 2..5) and 1.717, 1.702, 1.596, 1.649 (C = 2..5).
