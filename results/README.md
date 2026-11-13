# results/ - hardware and simulation outputs

Everything here was **copied** from the original working folders, not recomputed, except the
files marked "added by the reproduction code". Hardware runs were
done on IBM Quantum `ibm_phoenix` (120 qubits, square lattice, CZ) and `ibm_fez` (156 qubits, heavy
hex, CZ) with Qiskit Runtime `SamplerV2` (qiskit 2.5.1, qiskit-ibm-runtime 0.48.0, default options:
no twirling, no dynamical decoupling), transpiled with the identity layout
(`initial_layout=range(n)`), `optimization_level=1`, `seed_transpiler=0`, no SWAPs.

**raw** = what the device or simulator returned (counts, bit arrays, job metadata, the submitted
circuits, calibration). **derived** = computed from raw data (reconstructions, metrics, tables).

Checksums: `MANIFEST.sha256`. The mapping of each file to its original path and md5 was kept by
the author and is not part of the repository. Sanitisation: IBM instance CRNs and one billing-metric
UUID were replaced by `"<redacted>"`; token fields were already `"<removed>"`. Text in JSON/TOML was
edited only by these string replacements, so all files still parse.

## Conventions used by every file

* **Count keys** (`{bitstring: count}`) are Qiskit bitstrings: the **rightmost** character is
  classical bit 0. All whole-device circuits end with `measure_all()` (register `meas`, one bit per
  qubit) and are transpiled with the identity layout, so character `-(q+1)` (from the right) is
  physical qubit `q`. Python: `bits = key[::-1]; bit_of_qubit_q = int(bits[q])`.
* Data qubits are measured in the basis named in the file (`z`, `x`, `y`): `x` appends `H`, `y`
  appends `Sdg, H` to the data qubits only; address qubits are always measured in Z.
* **QPY files are gzip-compressed** (`*.qpy.gz`, gzip level 9, no stored name/mtime) to keep the
  repository small (10x-20x smaller). Load with
  `qiskit.qpy.load(gzip.open(path, "rb"))` (tested with qiskit 2.6.2; written by qiskit 2.5.1).
  They are the transpiled ISA circuits exactly as submitted, in job order. Circuit names inside the
  QPY are auto-generated (`circuit-48`, ...) and carry no meaning; use the order.
* `*.json.gz` files are gzip-compressed JSON (`json.load(gzip.open(p))`).
* NPZ files contain only numeric arrays (`np.load(p)` works with `allow_pickle=False`).
* Images are 0..1 floats, row-major, `[row, column]` with row 0 at the top.
* Pixel-art images inside JSON are base64 of raw RGBA uint8, row-major:
  `np.frombuffer(base64.b64decode(s), np.uint8).reshape(height, width, 4)`.
* DRR ("dynamic-range recovery", App. drr) method names: `none`, `min_max`, `percentile`,
  `mean_std`, `abs_max`, `min_avg_max`.
* **Address configurations without shots.** `tessa.decode_from_counts` returns the midpoint of
  the range (0.5 on [0, 1] in the single encoding) for an address configuration of a cell that
  received no shot, without a warning; `decode_from_counts(..., unsampled="nan")` returns NaN
  instead. Every configuration of the QCAT runs and of the coherence `z_image` circuits was
  sampled at least once (checked), so the stored reconstructions are not affected.

## Decoding the raw hardware data yourself

Each recipe uses only the files named and the public `tessa` API (install the repository with
`pip install -e ".[notebooks]"`; paths relative to the repository root). The `tessa_paper` function
that does the same, and compares the result with the paper, is given after each recipe.

**QCAT (Fig. 4, Table 1, Table 6).** Counts plus the stored encodings:

```python
import gzip, json
import numpy as np
from tessa import apply_dynamic_range, decode_from_counts, load_encoding

enc = load_encoding("results/hardware/qcat/encodings.json")["phoenix/single"]
counts = {
    "z": json.load(gzip.open("results/hardware/qcat/counts/phoenix_single_z.json.gz"))
}
raw = decode_from_counts(counts, enc)  # (28, 28) in [0, 1]
truth = np.load("results/hardware/qcat/reconstructions.npz")["phoenix/input_gray"]
drr = apply_dynamic_range(raw, truth, "mean_std")  # value only, no clipping
print(np.abs(raw - truth).mean(), np.abs(drr - truth).mean())  # 0.130 and 0.075 (Table 1)
```

Pair runs: `counts = {"z": ..., "x": ...}` from `<dev>_pair_{z,x}.json.gz`; `decode_from_counts`
then returns `(value, hue)`. Same in `tessa_paper.qcat.rederive_from_counts()`.

**Pixel art (Fig. 6).** `python -m tessa.hardware decode results/hardware/pixel_art/PixelArt --out
decoded` rebuilds `decoded/results/<label>/result.json` from `manifest.json`, `requests/` and
`job/result.json` (needs the `ibm` extra). Same in `tessa_paper.pixel_art.redecode_check()`.

**Coherences (Fig. 9).** The pair assignment is in `design.json`; the decode is
`tessa_paper.coherence.rederive_pairs_from_counts()` (K = 2 P(address bit = 0) - 1 per pair, from
`counts/phoenix.json.gz`).

**Genetic sequences (Table 8).** One ticket and one shot file per job, plus the stored
constellation point sets. The paper's "angle" encoding is the `levels` scheme here.

```python
import json
import numpy as np
from qiskit.primitives import BitArray
from tessa import encode_symbols, graph_from_coupling, symbols_from_counts

folder = "results/hardware/genetic/"
stem = "genetic_sequences_db91e1694fic73eua10g"  # or genetic_second_db980g0c77vs73811r9g
ticket = json.load(open(folder + stem + ".json"))
shots = np.load(folder + stem + ".npz")
points = np.load(folder + "constellation_points.npz")
graph = graph_from_coupling(
    ticket["edges"], ticket["n_qubits"]
)  # default colouring: 792 slots

# Sequence: the record of data/sequences/genetic_sequences.txt with the ticket's key
# (lines after its ">KEY ..." header up to the next header, joined), as alphabet indices.
key, scheme = "SHH_HUMAN", "constellation"  # run 2: see ticket["items"]
alphabet = ticket["alphabets"][key]
text, seq = open("data/sequences/genetic_sequences.txt").read().split("\n"), []
start = next(i for i, line in enumerate(text) if line.startswith(f">{key} "))
for line in text[start + 1 :]:
    if line.startswith(">"):
        break
    seq += [alphabet.index(c) for c in line.strip().upper()]
truth = np.array(seq)

per_slot = 1 if truth.size <= graph.capacity else 2  # one or two symbols per slot
_, enc = encode_symbols(
    graph,
    truth,
    len(alphabet),
    scheme=scheme,
    slot_order=ticket["slot_order"],
    points=points[f"points_{len(alphabet) ** per_slot}"]
    if scheme == "constellation"
    else None,
)
counts = {}
for i, (k, s, basis) in enumerate(ticket["circuits"]):  # job order: [key, scheme, basis]
    if (k, s) == (key, scheme):
        counts[basis] = BitArray(
            shots[f"circuit_{i}"], int(shots[f"num_bits_{i}"])
        ).get_counts()
for drr in ("none", "min_max", "min_avg_max"):
    got = symbols_from_counts(counts, enc, drr=drr)
    print(drr, round(100 * np.mean(got == truth), 1))  # 73.4, 73.2, 76.2 (Table 8, SHH)
```

`encode_symbols` chooses one symbol per slot when the sequence fits in the 792 slots and two
otherwise (`per_slot`); the measured bases are `enc.bases` (`z`, `x`, `y` for the constellation;
`z`, or `z` and `x`, for levels). Lines starting with `;` in the sequence file are comments. Same,
for both runs, every sequence and every DRR method, in `tessa_paper.genetic.hardware_results()`.

## Paper item -> files

| paper item | files (relative to `results/`) | notes |
|---|---|---|
| Fig. teaser, top (QCAT, Phoenix, Z) | `hardware/qcat/reconstructions.npz` (`phoenix/input_gray`, `phoenix/single/hardware/none`), `hardware/qcat/stats.json` | crop of Fig. qcat |
| Fig. teaser, bottom (City-pop cat) | `hardware/pixel_art/PixelArt/requests/citypopcat.json`, `.../results/citypopcat/result.json` | crop of Fig. pixelart |
| Fig. qcat, Table hw | `hardware/qcat/stats.json` (`stats`), `hardware/qcat/reconstructions.npz` | the paper's analysis (value-only DRR, no clipping); re-derivable from `hardware/qcat/counts/` + `hardware/qcat/encodings.json` and `simulation/qcat/` |
| CZ counts (Sec. hardware), Table prior (Tessa rows: qubits, values, CZ, 2Q depth) | `hardware/qcat/stats.json` (`circuits`), `hardware/qcat/circuits/phoenix_*.qpy.gz` | the 2Q depths printed in Table prior (19/39) do not match the circuits (23/43), see REPRODUCIBILITY |
| Fig. pixelart (a), (b) | `hardware/pixel_art/PixelArt/`, `hardware/pixel_art/Pixels2/` (`requests/*.json`, `results/*/result.json`) | raw: `job/result.json` |
| App. coherences, Fig. coherences | `hardware/coherence/tables/phoenix_B_pairs.csv`, `hardware/coherence/stats.json` (`devices.phoenix.B`) | the figure shows experiment B (pixel pairs), Phoenix |
| Table kcompare | none | no data or code survives; recomputed by the notebook |
| Fig. entmap, Fig. arealaw, Table pairs | `simulation/entanglement/{entanglement_map,area_law,pairs}.npz`, `summary.json` | reference copies written by `tessa_paper.entanglement` (noiseless statevector; the original script stored nothing) |
| Fig. pairclamp | `simulation/pair_encoding/pair_encoding_results.json` (`e2_clamp`) | |
| Table drr, image columns | `hardware/qcat/stats.json`: `stats["<dev>/single"]["hardware"][<method>]["gray_image"]["mae"]` | |
| Table drr, pitch columns; Table midi; Fig. sound | `simulation/midi_audio/midi_fakefez.json`, `wavetable_fakefez.npz`, `README.md` | **simulation** (FakeFez noise model) written by `tessa_paper.audio_midi`; it reproduces every printed number, which the paper labels as hardware; no hardware data survive |
| Fig. genetic-results, Table genetic, macros of App. genetic | `hardware/genetic/*`, `simulation/genetic/metrics.json` | the paper's LaTeX inputs to compare with are `figures/paper/results_{table,macros}.tex` |
| "32768 shots is the smallest power of two with zero noiseless constellation errors" | `simulation/genetic/shot_sweep.json` | |
| Fig. hh, Fig. circuits, Fig. clampregion, Table modes, Fig. constellation | none | drawn from device graphs / analytic, no stored results |

---------------------------------------------------------------------------------------------------

## hardware/qcat/ - QCAT single (Z) and pair (Z,X) runs, 2026-09-26

| job key | job id | backend | bases | shots | submitted (UTC) | input |
|---|---|---|---|---|---|---|
| `phoenix/single` | `das2d8bojkfs738opjmg` | ibm_phoenix | z | 8192 | 2026-09-26 20:10:09 | `data/images/qcat/qcat28.png` |
| `phoenix/pair` | `das2d8rojkfs738opjng` | ibm_phoenix | z, x | 8192 | 2026-09-26 20:10:11 | `data/images/qcat/qcat28_color.png` |
| `fez/single` | `das2d93g95ks73ef0bu0` | ibm_fez | z | 8192 | 2026-09-26 20:10:12 | `data/images/qcat/qcat.png` |
| `fez/pair` | `das2d9jojkfs738opjp0` | ibm_fez | z, x | 8192 | 2026-09-26 20:10:14 | `data/images/qcat/qcat_color.png` |

Encoding: single = theta = pi*gray, clamp 0, value range (0,1); pair = HSV value -> theta (Z,
clamp 0.2*pi), hue -> phi (X, half circle, clamp_b 0); unused slots hold 0.

| file | raw/derived | content and schema |
|---|---|---|
| `jobs.json` | raw (job metadata) | `{"<dev>/<mode>": {job_id, backend, bases[list], shots, submitted (ISO), sampler_options{environment{job_tags}}, status, metrics{caller, qiskit_version, timestamps{created, running, finished}, usage{qpu_charge_time_seconds, value, unit, status, details[{value, unit, metric{id: "<redacted>", value}}]}, circuits_execution_time_ns}, usage (QPU s), shots_seen{basis: n}, distinct_bitstrings{basis: n}}}`. `fez/pair` lacks the last two keys (they are in `stats.json`). |
| `calibration.json` | raw (summary only) | `{"phoenix"|"fez": {backend, num_qubits, two_qubit_gate, last_update, two_qubit_error, readout_error, sx_error, t1_us, t2_us: {median, mean, min, max, n}, backend_rates{sx_error, cz_error, readout_error, sx_duration [s], cz_duration [s], t1 [s], t2 [s]}}}`. Snapshot of `backend.properties()` at submission (last_update phoenix 2026-09-26T21:51:19+02:00, fez 21:52:08+02:00). **Only summary statistics were saved**; per-qubit/per-coupler calibration of these runs is lost. |
| `circuits/<dev>_<mode>.qpy.gz` | raw (submitted circuits) | transpiled circuits; `single`: 1 circuit (z); `pair`: 2 circuits (z, x). CZ per circuit: phoenix single 728, phoenix pair 1456, fez single 428, fez pair 852. |
| `counts/<dev>_<mode>_<basis>.json.gz` | raw (counts) | `{bitstring: count}`, 120 (phoenix) or 156 (fez) characters, 8192 shots; files: `phoenix_single_z`, `phoenix_pair_z`, `phoenix_pair_x`, `fez_single_z`, `fez_pair_z`, `fez_pair_x`. Original names were `<dev>_<mode>_hardware_<basis>.json.gz`. |
| `encodings.json` | derived (encoding description) | written by the library (converted from the original `encodings.pkl`, which is not shipped); see the library documentation for its schema. With `counts/` it re-derives the `hardware/none` reconstructions (`decode_from_counts`). |
| `stats.json` | derived (**the paper's analysis**) | byte-for-byte the analysis of 2026-09-26T20:12:03Z (value-only DRR, no clipping; `settings.drr_hue = false`), except `instance_crn` and `metric.id` redacted. `tessa_paper.qcat` recomputes every number of Table 1 and Table 6 used from the images and checks it against this file (`qcat.stats_crosscheck()`, differences below 1e-14). Keys: `settings` {generated, run_on_hardware, shots_per_setting 8192, clamp_over_pi{single 0, pair 0.2}, clamp_b_over_pi 0, value_range [0,1], figure_drr "mean_std", single_payload, pair_payload, colour_render, drr_scope, drr_hue false, simulation, sim_seed 1, instance_crn "<redacted>", versions{...}}; `calibration` (= `calibration.json`); `circuits["<dev>/<mode>"]` {bases, pixels, cells_used, fields, capacity, clamps_over_pi, logical_depth, logical_two_qubit_gates, transpiled_depth{basis}, transpiled_two_qubit_gates{basis}, swaps{basis}}; `jobs` (= `jobs.json` plus `fez/pair` shots_seen/distinct_bitstrings); `stats["<dev>/<mode>"][source][method]` with source in `hardware`, `local_sim`, `noiseless` and method in the six DRR names. Single mode: `{"gray_image": M}`; pair mode: `{"value_Z": M, "hue_X": H, "hue_X_if_corrected": H, "colour_rgb": C}`. `M` = {mae, rmse, max_abs_err, median_abs_err, bias, pearson_rho, psnr_db, ssim, contrast_ratio, fit_slope, fit_intercept, frac_err_gt_0.05, frac_err_gt_0.10, frac_err_gt_0.25, n}; `H` (hue errors on the circle, as a fraction of a full turn) = {mae, rmse, max_abs_err, median_abs_err, frac_err_gt_*, n}; `C` (rendered RGB) = {mae, rmse, max_abs_err, frac_err_gt_0.10, frac_err_gt_0.25, n}. Errors are on [0, 1]. Table hw: value MAE = `...[src]["none"]["gray_image"|"value_Z"]["mae"]`, DRR = same with `mean_std`, rho = `pearson_rho` (method none), hue MAE = `hue_X.mae` (none). |
| `reconstructions.npz` | derived (**the paper's analysis**) | 76 float64 arrays. `<dev>/<mode>/<source>/<method>`: single -> shape (H, W); pair -> (2, H, W) = (value, hue), hue as a fraction of the circle in [0, 1); `<dev>/input_gray` (H, W); `<dev>/input_value_hue` (2, H, W). H = W = 28 (phoenix) or 21 (fez). Read by the Fig. qcat script together with `stats.json`. |

Not shipped: the later re-analysis of 2026-09-27 (`stats.json`/`reconstructions.npz` in the original
`hardware/` folder, hue-DRR and clipping on). No paper item uses it; its DRR numbers differ from
Table hw / Table drr. Also not shipped: `stats.txt` (text rendering of `stats.json`), notebook PNG/PDF
panels, the offline FakeFez/FakeNighthawk run, `encodings.pkl` (pickle; replaced by `encodings.json`).

## simulation/qcat/ - local-noise and noiseless simulations of the same encodings

Produced on 2026-09-26 by the acquisition notebook: per super-pixel (bare cell: the data qubit and
its address qubits only, `halo=None`), 8192 shots per basis, seed 1. `local_sim` used a noise model
built from the live backend's per-qubit calibration at that time (Aer, `calibrated_sampler(backend,
local=True)`); that per-qubit calibration was not saved, so the local-noise run cannot be repeated
exactly. `noiseless` = shot noise only (`aer_sampler(graph, seed=1)`) and re-runs bit-exactly.

| file | raw/derived | content and schema |
|---|---|---|
| `counts/<dev>_<mode>_local_sim.json.gz` | raw (simulated counts) | `{site: {basis: {local_bitstring: count}}}`; `site` (string) = the data qubit index of the cell (equal to `DataGroup.site`); one entry per data qubit (phoenix 60, fez 64). The local bitstring has `1 + degree` characters; rightmost = the data qubit, then the cell's address qubits in the order of `address_qubits` of that group (in `hardware/qcat/encodings.json`), i.e. compact register `[data, *addresses]`, Qiskit bit order. 8192 shots per basis. |
| `sim.npz` | derived (reconstructions) | 12 float64 arrays `<dev>/<mode>/<local_sim|noiseless>/<j>`, j = field index (0 = gray or value, 1 = hue), shape (28, 28) or (21, 21), raw decode (no DRR). The DRR-processed versions and metrics are in `hardware/qcat/reconstructions.npz` and `stats.json` (sources `local_sim`, `noiseless`). |

## hardware/pixel_art/ - the two pixel-art hardware packages (Fig. pixelart)

| package | job id | backend | circuits | submitted -> finished (UTC) | runs (label: first circuit, n circuits, shots) |
|---|---|---|---|---|---|
| `PixelArt` | `dbbpav9gv2cc73e8u8ig` | ibm_phoenix | 28 | 2026-10-20 16:23:26 -> 20:10:10 (ran 20:08-20:09) | anime 0/3/8192, sleepingCat 3/3/8192, cat_transition 6/5/8192, apple_moth_gray 11/5/8192, citypopcat 16/3/8192, statue 19/3/8192, spaceship 22/1/8192, apple_moth_color 23/5/**16384** |
| `Pixels2` | `dbcdje7i0auc73duo9ag` | ibm_phoenix | 6 | 2026-10-21 15:26:48 -> 16:48:41 | ramen 0/3/8192, forest 3/3/8192 |

Label -> paper name: anime = Portrait, sleepingCat = Sleeping cat, citypopcat = City-pop cat,
cat_transition = QCAT -> dead QCAT, apple_moth_gray / apple_moth_color = Apple -> Moth (gray /
palette). Inputs: see `data/README.md`. The copy used is the one the paper's figure was made from,
whose `package.toml` labels the `Dias.png` run `sleepingCat` (an earlier working copy of the
package file said `waterfall`; manifest and results say `sleepingCat`).

Produced by the hardware-package tool (`tessa.hardware`; an earlier version of the same tool made the runs):
`run` builds every circuit, transpiles, submits one SamplerV2 job, fetches the result; `decode <folder>`
rebuilds `results/` and `summary.*` offline from `manifest.json`, `requests/` and `job/result.json`
(verified to reproduce every stored `result.json` exactly). The figure script needs only
`requests/<label>.json` and `results/<label>/result.json`.

| file | raw/derived | content and schema |
|---|---|---|
| `package.toml` | input record | the package as submitted: top-level defaults (`name, device, shots, mapping, seed`), `[[runs]]` (`label, image`/`images, representation, basis, mapping, seed, colorspace, transition_values, alphas, shots`), `[ibm] token = "<removed>", instance = "<redacted>"`. Image paths are relative to the original location (`../../pixel/<original name>`), see `data/README.md` for the mapping. |
| `manifest.json` | raw (job record) | {package, fingerprint (sha256 of the requests), tag (job tag `tessa-<name>-<uuid8>`), created, state, device, backend, instance "<redacted>", job_id, submitted, finished, status, n_circuits, depths[n_circuits] (transpiled depth), edges[[a,b]...] (218 Phoenix couplers), n_qubits 120, runs[{label, images (original file names), request_file, first, n_circuits, shots}], history[{time, status}], usage_estimation{quantum_seconds}} |
| `requests/<label>.json` | input record | the full request as the web page would send it: {shots, mapping, seed, representation, basis or transition_values + alphas, colorspace (if any), device, images[{width, height, rgba}]}. **`rgba` embeds the input image as base64 RGBA** (needed by the figure script for the input panels). |
| `circuits.qpy.gz` | raw (submitted circuits) | all transpiled circuits in job order (PixelArt 28, Pixels2 6), 120 qubits each, `measure_all`. |
| `calibration_submit.json.gz` | raw | full `backend.properties().to_dict()` of ibm_phoenix at submission: {backend_name, backend_version, last_update_date, qubits[120][{name, date, unit, value}], gates[...], general, general_qlists}. last_update 2026-10-20 17:47:45+02:00 (PixelArt), 2026-10-21 17:08:28+02:00 (Pixels2). The original `job/calibration_run.json` (calibration fetched with the result) is byte-identical and is not shipped. |
| `job/result.json` | raw (all shots) | the `PrimitiveResult` serialised with `qiskit_ibm_runtime.RuntimeEncoder`; read with `json.loads(text, cls=qiskit_ibm_runtime.RuntimeDecoder)`. `pub_results[i].data.meas` is a `BitArray` (120 bits, shots of circuit i); `metadata.execution.execution_spans` gives the data slices. Uncompressed because the decode tool reads it directly. |
| `job/metrics.json` | raw | IBM job metrics: {caller, qiskit_version, timestamps{created, running, finished}, usage{qpu_charge_time_seconds (8 / 4), value, unit, status, details}, circuits_execution_time_ns} |
| `results/<label>/result.json` | derived (decoded) | {representation, basis, width, height, n_pixels, capacity, slots_used, mapping, groups[60]{data, addresses[], n_values}, pixel_group[n_pixels], pixel_address[n_pixels], pixel_part[n_pixels] (which slot/part each opaque pixel uses), then either `image` (base64 RGBA, raw reconstruction) or `frames[5]` (base64 RGBA, one per readout angle) + `alphas_degrees` [0, 22.5, 45, 67.5, 90] + `transition_values` (+ `transition_palette`, list of the 16 shared [R, G, B] colours, palette transition only), `palette` (list of [R, G, B], dark to light; palette runs only), `metrics{...}` (raw), `drr{method (best DRR by its score), metrics{...}, scores{method: score}, image or frames}`, `run{kind "hardware", device, job_id, shots, first_circuit, n_circuits}`, `settings{...}`}. Metric keys: gray `mean_abs_error_255`; huevalue `mean_abs_error_value_255`, `mean_abs_error_hue_degrees`; color `mean_abs_error_rgb_255` (+ `_R_`, `_G_`, `_B_`); palette `palette_size`, `pixels_exact` (fraction), `mean_index_error`; transitions `mean_abs_error_A_at_0_255`, `mean_abs_error_all_angles_255` (against the noiseless frames), `mean_abs_error_B_at_90_255`, palette transition `pixels_exact_A_at_0`, `pixels_exact_B_at_90`, `mean_index_error_all_angles`. `_255` = on the 0-255 scale (the figure divides by 255); hue in degrees (the figure divides by 360). |
| `summary.csv` | derived | one row per run: settings and every `raw.*` / `drr.*` metric (columns padded with spaces; strip them when parsing). |

Dropped from the packages: `log.txt` (contains an absolute local path), `inputs/` (duplicates of
`data/images`), `job/calibration_run.json` (identical to `calibration_submit.json`), `summary.json`
(same rows as `summary.csv`), the rendered `raw*.png` / `drr*.png` / `*_x8.png` (re-created by
`decode`), `results/apple_moth_{color,gray}.zip` (zips of those PNGs), `.DS_Store`.

## hardware/genetic/ - DNA and protein sequences on ibm_phoenix, 2026-10-16

| run (ticket) | job id | submitted -> finished (UTC) | circuits | contents |
|---|---|---|---|---|
| `genetic_sequences_db91e1694fic73eua10g` | `db91e1694fic73eua10g` | 12:22:28 -> 12:28:45 | 15 | constellation scheme for POLG2, SUR1, pikachurin, SHH, p53 (Z, X, Y each) |
| `genetic_second_db980g0c77vs73811r9g` | `db980g0c77vs73811r9g` | 19:51:29 -> 21:16:38 | 16 | levels ("angle") scheme for all 7 sequences (Z only for the one-per-slot sequences MT-CO3, RRM1, SHH, p53; Z and X for the two-per-slot POLG2, SUR1, pikachurin) + constellation for MT-CO3 and RRM1 (Z, X, Y) |

32768 shots per circuit. The graph (120 qubits, 60 data + 60 address, 792 slots) was built from the
Phoenix calibration snapshot of 2026-10-09 11:17:07+02:00 shipped with the library
(`tessa.hardware` devices `ibm_phoenix.json`); slots were filled best-qubit first (`slot_order`).

| file | raw/derived | content and schema |
|---|---|---|
| `<run>.json` (ticket) | raw (job record) | {job_id, prefix (run 2 only: "genetic_second"), backend, submitted, status, keys + schemes (run 1) or items[[key, scheme]] (run 2), alphabets{key: symbols in index order}, shots 32768, circuits[[key, scheme, basis]] in job order, slot_order[792] (the order in which the 792 capacity slots were filled, best-measured qubits first), edges[[a,b]] (218 couplers), n_qubits 120, finished, shots_file} |
| `<run>.npz` | raw (all shots) | `circuit_<i>`: uint8 (32768, 15) = the packed bits of circuit i exactly as `qiskit.primitives.BitArray.array` (15 bytes = 120 bits per shot, big-endian byte order; the least significant bit of the last byte is classical bit 0 = qubit 0); `num_bits_<i>`: int scalar 120. Counts: `BitArray(z[f"circuit_{i}"], int(z[f"num_bits_{i}"])).get_counts()`. 15 (run 1) or 16 (run 2) circuits. |
| `constellation_points.npz` | encoding metadata (added by the reproduction code, 2026-11-13) | `points_4`, `points_16`, `points_20`, `points_400`: float64 arrays `(n, 3)` of unit Bloch vectors, the constellation point sets of the constellation runs (DNA and amino acids, one or two symbols per slot: point `j = s1 * k + s2` for two symbols, `j = s` for one). Decoding the shots with these sets reproduces every hardware cell of Table genetic. They were computed once with `tessa.constellation(n)` (Linux x86_64, numpy 2.5.4); `tessa.constellation` uses a short repulsion iteration whose last digits depend on the platform, and two hardware cells change when its start points move by a relative 1e-10, so `tessa_paper.genetic` reads these sets instead of recomputing them (`hardware_results(regenerate_constellation=True)` recomputes them). The circuits submitted for the genetic runs were **not saved**, so it cannot be checked which point sets were encoded on the chip; these are the sets with which the stored shots decode to the published table. |

## hardware/coherence/ - adjacent-pixel coherences (App. "What do the coherences mean?"), 2026-10-02

| device | job id | circuits | shots | submitted (UTC) | input |
|---|---|---|---|---|---|
| phoenix | `davnlrlj371s73dn89o0` | 30 | 8192 | 2026-10-02 09:35:43 | `data/images/qcat/qcat28.png` |
| fez | `davnls2l7guc73cf7vf0` | 18 | 8192 | 2026-10-02 09:35:45 | `data/images/qcat/qcat.png` |

Circuit names (job order, = order in `circuits/<dev>.qpy.gz` and keys of `counts/<dev>.json.gz`):
`k_image`, `k_reference`, `z_image`, `pairs_reference`, `pairs_0` ... `pairs_25` (phoenix) /
`pairs_13` (fez). Experiment A: `k_image` = the QCAT encoding (theta = pi * pixel) followed by `H` on
every address qubit, `K_a = 2 P(z_a = 0) - 1`; `k_reference` = the same with a low-contrast payload
(scale 0.01); `z_image` = the plain Z readout. Experiment B (the one in Fig. coherences): every
horizontally and vertically adjacent pixel pair (i, j) is encoded as an isolated degree-one
super-pixel (one data + one address qubit) on a maximum matching of disjoint low-error couplers
(phoenix 60 couplers, 1512 pairs in 26 circuits; fez 64 couplers, 840 pairs in 14 circuits), then
`H` on the address qubit; `K = 2 P(address = 0) - 1`, exact value `cos(pi (v_i - v_j) / 2)`;
`pairs_reference` = every coupler with values (0.5 + 0.005, 0.5 - 0.005). Pair order and the
assignment of pairs to circuits and couplers (seed 7) are in `design.json`.

| file | raw/derived | content and schema |
|---|---|---|
| `jobs.json` | raw | `{dev: {job_id, backend, circuits[names in job order], shots, submitted, sampler_options, status, metrics{...}, usage}}` (as in `hardware/qcat/jobs.json`) |
| `calibration.json` | raw (summary only) | `{dev: {backend, num_qubits, last_update, two_qubit_error, readout_error, t1_us, t2_us: {median, mean, min, max, n}, backend_rates{...}}}`; last_update phoenix 2026-10-02T10:33:21+02:00, fez 11:17:06+02:00 |
| `circuits/<dev>.qpy.gz` | raw | the transpiled circuits as submitted, job order |
| `counts/<dev>.json.gz` | raw | `{circuit_name: {bitstring: count}}`, whole-device bitstrings (120 / 156 chars), 8192 shots each. Original name `<dev>_hardware.json.gz`. |
| `design.json` | derived (experiment design) | written by the library (converted from the original `design.pkl`, not shipped): address list, matching, pairs, batches, exact K values. See the library documentation. |
| `sim.npz` | derived (local-noise simulation) | float64 arrays, all **probabilities P(measured bit = 0)** (convert with `K = 2P - 1`): `<dev>/k_image`, `<dev>/k_reference` (one per address qubit, ascending address-qubit index; phoenix 60, fez 92; light-cone cuts), `<dev>/pairs` (one per pixel pair, pair order = all horizontal pairs row-major, then all vertical pairs row-major; 1512 / 840), `<dev>/pairs_reference` (one per coupler of the matching; 60 / 64); and `<dev>/z_image` (784 / 441 reconstructed pixel values in [0,1], flattened row-major). Local per-qubit calibrated noise (Aer), seed 1, 8192 shots. |
| `stats.json` | derived | {`settings` {generated 2026-10-02T09:42:45Z, shots_per_circuit 8192, encoding, observable, reference_scale 0.01, pair_seed 7, pair_reference_delta 0.005, weak_reference_threshold 0.5, classical_route_drr, simulation, sim_seed, instance_crn "<redacted>", versions}, `jobs` (= jobs.json), `devices[dev]` {device, calibration, circuits[name]{depth, two_qubit_gates, swaps}, `A` {n_address_qubits, exact_image{mean,min,max}, exact_reference_min, hardware/local_sim {metrics{measured, classical, classical_drr: {n, mae, rmse, bias, pearson_rho, fit_slope}}, metrics_healthy{...}, metrics_by_degree{degree: {...}}, standard_error, probe, ...}}, `B` {couplers, pairs, circuits, flat_pairs, exact_reference, hardware/local_sim {metrics{all_pairs, healthy_pairs, horizontal, vertical, combined: {n, mae, rmse, bias, pearson_rho, fit_slope}}, standard_error{mean,max}, reference_summary{mean_K, min_K, weak_couplers, pairs_on_weak_couplers}}}}}. Fig. coherences numbers: `devices.phoenix.B.hardware.metrics.{horizontal,vertical,combined}` (MAE 0.022 / 0.019 / 0.020, rho 0.977 / 0.990 / 0.988). |
| `tables/<dev>_B_pairs.csv` | derived | one row per pixel pair: direction (`h`/`v`), pixel_i, pixel_j ("row,col"), value_i, value_j, data_qubit, address_qubit, K_exact, hardware_measured (K), hardware_measured_se, hardware_coupler_reference (K of the reference circuit on that coupler), local_sim_measured, local_sim_measured_se, local_sim_coupler_reference. Enough to redraw Fig. coherences without the design file. |
| `tables/<dev>_A_addresses.csv` | derived | one row per address qubit: address_qubit, degree, x, y (grid position), K_exact, K_exact_reference, t1_us, t2_us, readout_error, then for `hardware_` and `local_sim_`: healthy (reference K >= 0.5), measured, measured_se, classical (Eq. Ka on the Z reconstruction), classical_se (nan), classical_drr, classical_drr_se (nan), reference, reference_se. |

Not shipped: `design.pkl` (pickle of library and qiskit objects), `stats.txt`, the notebook figures
(`edges_*`, `coherence_scatter`, `coherence_chip_maps`; Fig. coherences itself is
`figures/paper/coherences.png`), the offline fake-backend run.

## simulation/pair_encoding/

`pair_encoding_results.json` (derived; written by the original analysis script of the paper, not
published, experiments `e1`-`e4`). Keys: `e1_exact` {device ibm_fez (FakeFez), n_qubits 156, capacity 448, and for `single`,
`pair`, `pair_clampb0.1pi`, `pair_full`: {values, clamps_over_pi, max_abs_error_per_field}} (exact
per-cell statevector check); **`e2_clamp`** (Fig. pairclamp) {shots 4096, lattice "4x4",
clamps_over_pi[9], single{rms_a, pred_a}, pair_cbc / pair_cb0 / pair_full: [9 x {clamp_over_pi, rms_a,
rms_b, pred_a, pred_b}]} (noiseless Aer, 3 random payloads, seeds 11-13; rms over payload values on
[0,1]; pred = analytic prediction); `e3_colour` {shots_per_setting 8192, device "FakeFez, local",
runs{fez_two_single, fez_pair, fez_pair_full, noiseless_pair_full: {value_mae, hue_mae, value_rho,
hue_outliers_gt_0.25, logical_depth, settings, mean_std_value_mae}}, recon{run: [value, hue] 21x21
lists]}} (input `data/images/qcat/qcat_color.png`; not shown in the paper); `e4_sign_bit`
{shots_per_setting 8192, clamp_over_pi 0.2, clamp_b_over_pi 0.2, bits 448, continuous_values 896,
noiseless / fez: {bit_error_rate, a_mae, b_mae_given_bit}} (not shown in the paper). This is the
copy the paper used; a later version of the script gives the same `e2_clamp` and a different
`e3_colour`.

## simulation/genetic/

| file | raw/derived | content and schema |
|---|---|---|
| `metrics.json` | derived (simulation) | {device "ibm_phoenix", calibrated_at "2026-10-09 11:17:07+02:00" (snapshot used for the noise model), shots 32768, seed 7, sequences{key: {alphabet, "<scheme> \| noiseless", "<scheme> \| phoenix noise": {best_drr, metrics{method: {accuracy, errors, first, second, "within 1", "mean dist"}}}}}} with scheme in `constellation`, `levels` (the paper's "angle"). accuracy/first/second are fractions (first/second = first and second half of two-per-slot sequences); "within 1" = fraction within one alphabet step; "mean dist" = mean index distance. Noiseless = Aer shot noise; "phoenix noise" = per-cell local calibrated noise from the snapshot. Source of the "no noise" and "simulated noise" columns of Table genetic. |
| `shot_sweep.json` | derived (simulation) | {key: {n, alphabet, per_slot, errors{"constellation": {shots (256 ... 65536, as string): [errors for seeds 1..5]}}}} for POLG2, SUR1, P53, pikachurin, SHH; noiseless. Backs the 32768-shot choice. |

`reconstructed.fasta` of the original outputs is not shipped (no paper item uses it).

Added by the reproduction code (2026-11-12):

| file | raw/derived | content and schema |
|---|---|---|
| `metrics_rerun.json` | derived (simulation, re-run) | `metrics.json` regenerated with `tessa_paper.genetic.rerun_simulation()` (tessa 1.0.0, qiskit 2.6.2, qiskit-aer 0.18.2, Linux aarch64), same schema plus a `note`. Identical to `metrics.json` except `MT_CO3` / `constellation \| phoenix noise`, where one of 784 symbols differs for the methods `none` (69 vs 70 errors) and `mean_std` (66 vs 67). Not used by the table, which reads `metrics.json`. |

## simulation/entanglement/ and simulation/midi_audio/

Reference outputs written by `tessa_paper.entanglement.compute_all()` and
`tessa_paper.audio_midi.compute_all()` (2026-11-12); schemas in the README of each folder.
`simulation/entanglement/`: `entanglement_map.npz`, `area_law.npz`, `pairs.npz`, `summary.json`
(exact statevector, bit-identical to the original script). `simulation/midi_audio/`:
`midi_fakefez.json`, `wavetable_fakefez.npz` (**simulation** with the FakeFez noise model; the
paper's "hardware" MIDI numbers are this simulation, and no hardware data for Table midi or Fig.
sound survive).
