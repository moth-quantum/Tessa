# simulation/midi_audio/ (Table midi, Table drr pitch columns, Fig. sound) - SIMULATION ONLY

Nothing measured on hardware backs these paper items. The paper labels Table midi and the pitch
columns of Table drr as ibm_fez hardware ("hardware, raw", "Fez score"), but every printed number
is reproduced by the FakeFez noise-model simulation of the original notebook
(an earlier analysis notebook of the author, not published): machine-average noise
from the FakeFez calibration of qiskit-ibm-runtime (`tessa.backend_rates`), each data qubit
simulated alone with its address qubits, 8000 shots, Aer seeds drawn per cell from
`default_rng(1)`, input `data/midi/telemann_small.mid` (347 notes, capacity 448). The files here
were written by `tessa_paper.audio_midi.compute_all()` (2026-11-12, tessa 1.0.0, qiskit 2.6.2,
qiskit-aer 0.18.2, qiskit-ibm-runtime 0.52.1; about 15 s). The original code, run from a copy
with the same packages, gives the same reconstructions to 2e-16 and every printed number.

| file | content |
|---|---|
| `midi_fakefez.json` | `kind` ("SIMULATION ..."), `input`, `capacity` 448, `shots` 8000, `seed` 1, `rates` (FakeFez averages: sx_error, cz_error, readout_error, sx_duration [s], cz_duration [s], t1 [s], t2 [s]), `runs{name: {n_notes, pitch_exact_fraction, pitch_mae_semitones, duration_mae_s}}` with names `noiseless` (exact statevector), `pitch_drr=<method>` (DRR on both fields, linear durations; Table drr pitch columns and the raw row of Table midi) and `mean_std, duration_warp=<linear|log|sqrt>` (Table midi recovered rows). |
| `wavetable_fakefez.npz` | Fig. sound re-creation: `sine` (448) = sin(2 pi k / 448), `noiseless` and `fakefez` (448) reconstructions in [-1, 1] (value range (-1, 1), same FakeFez simulation, 8000 shots, seed 1, raw decode; mean absolute error 0.289, as the original notebook printed). |

What is lost, and cannot be re-created without new hardware runs:

* The one real ibm_fez MIDI run (2026-07-29, 8192 shots, min_max DRR: 39.5 % exact pitch,
  1.62 st, 0.469 s) saved no counts and no job id; its numbers are not in the paper.
* Fig. sound: the ibm_fez wavetable (448-sample sine, 2026-07-29) saved no counts, job id or
  reconstructed table; only a rendered WAV survives (not shipped). The caption's "4096 shots" and
  "min_max DRR" contradict the notebook code (8192 shots, division by the maximum). The paper's
  image is `figures/paper/sound_example.png`; the notebook draws a labelled re-creation from
  `wavetable_fakefez.npz` (`figures/sound_example_recreation.png`).
