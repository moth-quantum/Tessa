# Credits and third-party data

The repository's licence (see `LICENSE`) covers the **code only**. The data listed below keep their
own terms.

## Pixel art (`data/images/pixel_art/`, `data/images/qcat/`, and the images embedded in `results/hardware/pixel_art/`)

Pixel art: Spaceship, Statue, Portrait and Ramen by M.A. Kozhikkal (SalvaPixel, https://salvapixel.com/).
City-pop cat, Apple and Moth by M. Fish (https://www.artstation.com/leviatic). Forest by D. Souza
(https://x.com/derekpxart). Sleeping cat by E. Kılınç (Diaspixels, https://diaspixels.artstation.com/).
QCAT by D. Bultrini. Dead QCAT by S. Cantori.

| picture | file(s) | artist | gallery |
|---|---|---|---|
| Spaceship | `data/images/pixel_art/spaceship.png` | M.A. Kozhikkal (SalvaPixel) | https://salvapixel.com/ |
| Statue | `data/images/pixel_art/statue.png` | M.A. Kozhikkal (SalvaPixel) | https://salvapixel.com/ |
| Portrait | `data/images/pixel_art/portrait.png` | M.A. Kozhikkal (SalvaPixel) | https://salvapixel.com/ |
| Ramen | `data/images/pixel_art/ramen.png` | M.A. Kozhikkal (SalvaPixel) | https://salvapixel.com/ |
| City-pop cat | `data/images/pixel_art/citypop_cat.png` | M. Fish | https://www.artstation.com/leviatic |
| Apple | `data/images/pixel_art/apple.png` | M. Fish | https://www.artstation.com/leviatic |
| Moth | `data/images/pixel_art/moth.png` | M. Fish | https://www.artstation.com/leviatic |
| Forest | `data/images/pixel_art/forest.webp` | D. Souza | https://x.com/derekpxart |
| Sleeping cat | `data/images/pixel_art/sleeping_cat.png` | E. Kılınç (Diaspixels) | https://diaspixels.artstation.com/ |
| QCAT | `data/images/qcat/qcat.png`, `qcat28.png` (and the viridis renderings `qcat_color.png`, `qcat28_color.png`) | D. Bultrini | none |
| Dead QCAT | `data/images/qcat/qcat_dead.png` | S. Cantori | none |

The third-party pictures (all except QCAT) are included **with the artists' permission**, for the
purpose of reproducing the paper. They remain the copyright of their artists and are **not** covered
by the repository's licence: do not reuse them outside this repository without the artist's
permission. The same applies to the pictures embedded (base64) in
`results/hardware/pixel_art/*/requests/*.json` and to the hardware reconstructions of them in
`results/hardware/pixel_art/*/results/*/result.json`, and to `figures/paper/fig_pixelart.pdf`.
If you show a picture or a result on its own, credit its artist and link to their gallery.

The paper credits them in the caption of Fig. pixelart ("Pixel art: Spaceship, Statue, Portrait and
Ramen by M.A. Kozhikkal (SalvaPixel); City-pop cat, Apple and Moth by M. Fish; Forest by D. Souza;
Sleeping cat by E. Kılınç (Diaspixels); QCAT by the author; dead QCAT by S. Cantori.") and in the
acknowledgements ("Additional non-QCAT pixel art was provided by M.A. Kozhikkal (SalvaPixel),
M. Fish, D. Souza and E. Kılınç (Diaspixels).").

## Genetic sequences (`data/sequences/genetic_sequences.txt`)

Fetched 2026-10-15 (p53, pikachurin and Sonic hedgehog added 2026-10-16).

* **UniProt** (UniProtKB/Swiss-Prot), licensed under CC BY 4.0
  (https://www.uniprot.org/help/license). The UniProt Consortium, "UniProt: the Universal Protein
  Knowledgebase in 2023", Nucleic Acids Res. 51, D523 (2023). Entries used by the paper:
  Q09428-2 (ABCC8 / SUR1, isoform 2), P23921 (RRM1), Q63HQ2 (EGFLAM / pikachurin), Q15465 (SHH),
  P04637 (TP53 / p53). Also present in the file but not used: Q7Z2Y5 (NRK), O95255 (ABCC6 / MRP6).
  The sequences are unmodified; the file adds a short key and a header line per record.
* **NCBI** RefSeq / GenBank (no use restrictions; https://www.ncbi.nlm.nih.gov/home/about/policies/).
  O'Leary et al., "Reference sequence (RefSeq) database at NCBI", Nucleic Acids Res. 44, D733 (2016).
  Records used by the paper: NM_007215.4 (POLG2 mRNA), NC_012920.1:9207-9990 (human mitochondrion,
  MT-CO3). Also present but not used: J01859.1 (E. coli 16S rRNA), NM_005684.5 (GPR52 mRNA).

## MIDI (`data/midi/`)

* `telemann_small.mid`: excerpt of G.P. Telemann, Fantasia No. 12 for solo flute (the composition is
  in the public domain). **Source of the MIDI transcription not recorded.**
* `invent15.mid`: J.S. Bach, Invention No. 15 (the composition is in the public domain).
  **Source of the MIDI transcription not recorded.**

No audio files are shipped.

## Hardware

The hardware results in `results/hardware/` were obtained on IBM Quantum devices `ibm_phoenix` and
`ibm_fez` through Qiskit Runtime; job ids are listed in `results/README.md`.
