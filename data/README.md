# data/ - source data, before encoding

Every file here is an input that a figure or table of the paper encodes. Files are byte-identical
copies of the originals (no re-encoding, no metadata stripping); `MANIFEST.sha256` lists their
checksums (the mapping to the original paths and md5 sums was kept by the author and is not shipped).
Credits and permissions: see `../CREDITS.md`. The third-party pixel art is **not** covered by the
repository's code licence.

Pixel values are read as 8-bit integers and divided by 255 before encoding unless stated otherwise.
"theta = pi * v" means the value v in [0, 1] is stored as the rotation angle of a data qubit.

## images/qcat/

| file | format | used by | preprocessing before encoding | origin |
|---|---|---|---|---|
| `qcat.png` | L (gray), 21x21, 338 B | Fig. qcat / Table hw (Fez, Z), Fig. coherences source for Fez (not shown), Fig. pixelart (b) QCAT transition (image A), Fig. entmap (App. entanglement) | `convert("L")/255` -> 441 values in [0,1], theta = pi*v, clamp 0; Fez capacity 448, 7 slots padded with 0. In the transition run the hardware-package pipeline (`tessa.hardware`) reads it as an RGBA request (luminance 0.299R+0.587G+0.114B) | QCAT by D. Bultrini (the author) |
| `qcat28.png` | L (gray), 28x28, 482 B | Fig. teaser (top), Fig. qcat / Table hw / Table prior (Phoenix, Z), Fig. coherences (Phoenix) | as `qcat.png`; 784 values, Phoenix capacity 792, 8 slots padded with 0 | QCAT by D. Bultrini |
| `qcat_color.png` | RGB, 21x21, 775 B | Fig. qcat / Table hw (Fez, Z,X); also the input of experiment e3 in `results/simulation/pair_encoding` (not shown in the paper) | `rgb/255` -> `matplotlib.colors.rgb_to_hsv`; V -> theta (Z readout, clamp 0.2*pi), H -> phi (X readout, half circle, clamp_b 0); S discarded (rendered at S=1). 882 values | `round(255*viridis(qcat.png/255))` (verified: max abs difference 0) |
| `qcat28_color.png` | RGB, 28x28, 1065 B | Fig. qcat / Table hw / Table prior (Phoenix, Z,X) | as `qcat_color.png`; 1568 values | `round(255*viridis(qcat28.png/255))` (verified, max abs difference 0). Recovered from the version history of the original (unpublished) code base, md5 4cfb9567... |
| `qcat_dead.png` | L (gray), 21x21, 340 B | Fig. pixelart (b) QCAT transition (image B, run `cat_transition`) | luminance /255 -> phi (half circle) of the transition encoding | Dead QCAT by S. Cantori |

`qcat28.png` here (md5 ca4874ce...) is the latest version of the file; an earlier version in the
original code base has a different md5 (9f7cb693...) but identical pixel values. Both equal `phoenix/input_gray` in
`results/hardware/qcat/reconstructions.npz`.

## images/pixel_art/

Inputs of the two pixel-art hardware packages (Fig. pixelart, Fig. teaser bottom row). Names were
normalised to lower case; the requests stored with the hardware results
(`results/hardware/pixel_art/*/requests/<label>.json`) embed the exact RGBA pixels that were encoded,
so the decode and the figure do not read these files. "opaque px" = pixels with alpha >= 128, the
only ones encoded.

| file | format | original file name | package / run label(s) | paper name, panel | encoding (representation, bases, mapping) | opaque px / values | artist |
|---|---|---|---|---|---|---|---|
| `spaceship.png` | RGBA 23x23 | `Spaceship.png` | PixelArt / `spaceship` | Spaceship, (a) col. 1 | palette (13 colours), Z, rectangles | 354 / 354 | M.A. Kozhikkal (SalvaPixel) |
| `statue.png` | RGBA 36x40 | `Statue.png` | PixelArt / `statue` | Statue, (a) col. 2 | gray, Z X Y, patches (seed 0) | 940 / 940 | M.A. Kozhikkal (SalvaPixel) |
| `sleeping_cat.png` | LA 130x91 | `Dias.png` | PixelArt / `sleepingCat` | Sleeping cat, (a) col. 3 | gray, Z X Y, lines | 1506 / 1506 | E. Kılınç (Diaspixels) |
| `forest.webp` | RGB (WebP) 39x39 | `waterfall.webp` | Pixels2 / `forest` | Forest, (a) col. 4 | gray, Z X Y, lines | 1521 / 1521 | D. Souza |
| `citypop_cat.png` | RGBA 28x28 | `Leviatic3.png` | PixelArt / `citypopcat` | City-pop cat, (a) col. 5 and teaser | huevalue (V in theta, H in phi; S from the input, not encoded), Z X Y, lines | 784 / 1568 | M. Fish |
| `portrait.png` | RGBA 23x23 | `AnimeGurl.png` | PixelArt / `anime` | Portrait, (a) col. 6 | color (RGB, 3 channels, 2 per slot), Z X Y, patches (seed 0) | 419 / 1257 | M.A. Kozhikkal (SalvaPixel) |
| `ramen.png` | RGBA 23x23 | `RamenBowl.png` | Pixels2 / `ramen` | Ramen, (a) col. 7 | color (HSV), Z X Y, lines | 368 / 1104 | M.A. Kozhikkal (SalvaPixel) |
| `apple.png` | RGBA 28x28 | `Leviatic1.png` | PixelArt / `apple_moth_gray`, `apple_moth_color` (image A) | Apple, (b) rows 2-3 | transition: A in theta; gray (lines) and palette of 16 colours (lines, 16384 shots) | 784 | M. Fish |
| `moth.png` | RGBA 28x28 | `levianc2.png` | PixelArt / `apple_moth_gray`, `apple_moth_color` (image B) | Moth, (b) rows 2-3 | transition: B in phi (half circle) | 784 | M. Fish |

The QCAT transition (`cat_transition`, PixelArt) uses `images/qcat/qcat.png` and
`images/qcat/qcat_dead.png`; they are not duplicated here. In the hardware packages these images were
copied to `inputs/` with hand-changed names (`anime.png`, `sleepingCat.png`, `citypopcat.png`,
`apple.png`, `moth.png`, `forest.webp`, `ramen.png`, `Spaceship.png`, `Statue.png`; all md5-identical
to the originals); the `inputs/` folders are not shipped because they duplicate this folder. The
image paths inside `results/hardware/pixel_art/*/package.toml` still point to the original
locations (`../../pixel/<original name>`, `../../data/qcat*.png`); use the table above to map them.

Preprocessing (inside the hardware-package pipeline, `tessa.hardware`): alpha < 128 -> not encoded;
gray = 0.299 R + 0.587 G + 0.114 B; palette indices sorted dark to light; pixels assigned to data
qubits by the mapping (lines / patches / rectangles); colour spaces as listed.

## midi/

| file | size | content | used by | preprocessing | origin |
|---|---|---|---|---|---|
| `telemann_small.mid` | 4211 B | 347 notes, single flute track, pitch 62-88, durations 0.133-2.998 s | Table midi, Table drr (pitch columns) | each note -> a pitch value and a duration value in [0, 1] (original code not published, ported as `tessa_paper.audio_midi`; the linear / log / sqrt duration warps are compared in Table midi), stored on the Fez graph (capacity 448) | Telemann, Fantasia No. 12 for solo flute (excerpt). Transcription source not recorded (MIDI header has only the track name "Flute, Flute" and an empty creator) |
| `invent15.mid` | 4299 B | 527 notes, 62.9 s | Fig. sound: the piece played with the quantum-reconstructed wavetable (the figure itself shows only the wavetable) | rendered note by note with the 448-sample wavetable; the wavetable input is a synthetic sine `sin(2 pi k / 448)`, not a file | J.S. Bach, Invention No. 15. Transcription source not recorded |

## audio/ (not shipped)

No audio file is an input of a paper item. Fig. sound encodes a synthetic 448-sample sine, generated
in code. `test_audio.aif` (11.6 MB, source not recorded) from the original repository is used only by
a neural-audio demo that is not part of the paper and is therefore not included.

## sequences/

| file | size | content | used by | preprocessing | origin |
|---|---|---|---|---|---|
| `genetic_sequences.txt` | 15075 B | FASTA-like, 11 records, header `>KEY type=<dna|protein> length=<n> source=<db:accession> \| original title`, fetched 2026-10-15/16 | Fig. genetic-results, Table genetic (and the macros of App. genetic) | symbols -> alphabet index (DNA `ACGT`; amino acids in Kyte-Doolittle hydropathy order `IVLFCMAGTSWYPHEQDNKR`); <= 792 symbols: one per slot; longer: two per slot (first half in position 1, second half in position 2); "constellation" or "levels" (= the paper's "angle") scheme | NCBI and UniProt, see `../CREDITS.md` |

Records used by the paper (7): `POLG2_mRNA` (NM_007215.4, 1582 nt), `SUR1_ABCC8_iso2`
(UniProt Q09428-2, 1582 aa), `MT_CO3` (NC_012920.1:9207-9990, 784 nt), `RRM1_HUMAN` (P23921, 792 aa),
`PIKACHURIN_HUMAN` (Q63HQ2, 1017 aa), `SHH_HUMAN` (Q15465, 462 aa), `P53_HUMAN` (P04637, 393 aa).
Records in the file but not used by any paper item (4): `ECOLI_16S_rRNA` (J01859.1), `GPR52_mRNA`
(NM_005684.5), `NRK_HUMAN` (Q7Z2Y5), `MRP6_ABCC6_HUMAN` (O95255). The file is shipped unmodified
because the genetic notebook reads it by key.
