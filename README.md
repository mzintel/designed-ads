# Designed ADs analysis code

Code and processed data for the designed activation domain study.

Sequences were generated with Markov chains trained on plant activation
domains, filtered through activation-domain predictors, and measured by
sort-seq in yeast. Selected designs were retested in plant leaves and in
HEK293T cells.

## Repository layout

| Folder | Contents |
|---|---|
| `designed AD generation notebooks/` | Markov chain sequence generation and predictor filtering used to choose the 18,750 tested designs |
| `sequencing data processing notebooks/` | Sort-seq read processing and activity calculation |
| `flow notebooks/` | Flow cytometry gating and per-plate processing for the HEK293T experiments |
| `LevenshteinDistance/` | Minimum edit distance of designed sequences to reference sequence sets |
| `notebooks/` | Sequence feature calculation, MMseqs2 clustering, predictor merging |
| `figures_notebooks/` | `Figures.ipynb` produces every main and Extended Data panel |
| `data/` | Inputs read by `Figures.ipynb` |
| `data_predictions/` | Predictor scores and ROC/PRC curve data |
| `supplementary_tables/` | Supplementary tables |

## Order of analysis

1. **Design.** `designed AD generation notebooks/4. Markov_Chain_Maker_Original.ipynb`
   generates sequences; `ADHunter, Paddle, TADA Predictor Analysis.ipynb` and
   `250 Combination Cleaning.ipynb` apply the predictors and select designs.
2. **Sequencing.** `labtools_SynADs_*.ipynb` convert reads to per-barcode
   activities; `CollapseADs_*.ipynb` average barcodes, harmonize against the
   controls library, and write per-sequence activities.
3. **Flow.** `Set_Gates_*.ipynb` define gate polygons, saved to a pickle.
   `Process_Plate_*.ipynb` apply them and write
   `data/HEK293T14_{EpCAM,ICAM1}_AD_ID.csv`. `GatingFigureS15*.ipynb` redraw
   the gating series for the supplement.
4. **Sequence analysis.** `notebooks/Calculate_Sequence_Features_500k.ipynb`
   (localCIDER), `Cluster_MMSeqs2_50pctidentity.ipynb`,
   `CombinePredictions.ipynb`, and `LevenshteinDistance/`.
5. **Figures.** `figures_notebooks/Figures.ipynb`.

## Key data files

| File | Contents |
|---|---|
| `data/500k_AA_SS_seqfeats.csv` | All 500,000 generated sequences with state size and localCIDER features |
| `data/500k_preds.csv` | All six predictor scores for the 500,000 generated sequences |
| `supplementary_tables/18750designed_activities_preds.csv` | Measured activities and predictions for the 18,750 tested designs |
| `data/Controls_supptable_allactivities.csv` | Controls library activities, used to harmonize across sorts |
| `data/allactivities_seqs_in_sort2.csv` | Per-sequence activities from the Sort 2 ratio sort |
| `data/SynAD_Glu_2_allinfo.csv` | Per-barcode reads and bin distributions |
| `data/MinSynth500K_to_*.csv` | Minimum edit distance to each reference set |
| `data/*_Yeast14byBCrep_AD_ID.csv` | Per-barcode yeast measurements for the 14 retested designs |
| `data/HEK293T14_{EpCAM,ICAM1}_AD_ID.csv` | HEK293T CRISPRa flow results |
| `data/Plant_Yeast14byBCrep_AD_ID.csv` | *N. benthamiana* leaf measurements |
| `data_predictions/consensus_model_all_six_predictions.tsv` | Six-predictor scores for 230,302 natural transcription factors |
| `data_predictions/*_roc_prc_data.csv` | Precomputed ROC/PRC curves per dataset and predictor |

### Large files hosted on Zenodo

Fifteen files exceed GitHub's file size limit and are archived on Zenodo
instead: **DOI: 10.5281/zenodo.XXXXXXX** (replace with the record DOI).

Download them and place each one back in the folder listed below before
running `Figures.ipynb`, which expects them at these relative paths.

| File | Size | Folder |
|---|---|---|
| `consensus_model_all_six_predictions.tsv` | 315 MB | `data_predictions/` |
| `500000_shuffled_ADpred_preds_dask.csv` | 263 MB | `data_predictions/` |
| `500k_AA_SS_seqfeats.csv` | 104 MB | `data/` |
| `500k_preds.csv` | 82 MB | `data/` |
| `500000_shuffled_53_ADhunter_ensemble_preds.csv` | 60 MB | `data_predictions/` |
| `500000_shuffled_53_Paddle_preds_dask.csv` | 45 MB | `data_predictions/` |
| `MinSynth500K_to_proteome.csv` | 43 MB | `data/` |
| `MinSynth500K_to_choptf.csv` | 43 MB | `data/` |
| `MinSynth500K_to_train.csv` | 43 MB | `data/` |
| `MinSynth500K_to_synth18750.csv` | 43 MB | `data/` |
| `MinSynth500K_to_self.csv` | 43 MB | `data/` |
| `500000_shuffled_TADA_preds_dask.csv` | 37 MB | `data_predictions/` |
| `500000_shuffled_ADHunter_preds_dask.csv` | 37 MB | `data_predictions/` |
| `synthetic_experimental_and_preds_w_parrot.csv` | 28 MB | `data_predictions/` |
| `500k_preds.csv` | 82 MB | `supplementary_tables/` (identical to the copy in `data/`) |

Without these files, `Figures.ipynb` will raise `FileNotFoundError` on the
data-loading cell. Every other notebook runs without them.

### Externally sourced data

These are redistributed from published work and should be cited to their
original sources, not to this repository:

- `data/Morffy_41586_2024_7707_MOESM2_ESM.csv` — Morffy *et al.* 2024, *Nature* 632:166–173 (supplementary data).
- `data/staller2018data.csv` — Staller *et al.* 2018.
- `data/sanborn_mutants.csv` — Sanborn *et al.* 2021, *eLife* 10:e68068.

## Running the figure notebook

`figures_notebooks/Figures.ipynb` reads from `../data/` and
`../data_predictions/` using relative paths, so it runs in place once the
Zenodo files above are downloaded into those folders. It saves PDFs to `../../Figures_v5/`, which is
outside this repository; change that path or create the folder before
running. Panels not included in the submission are saved with a
`NOTINCLUDED_` prefix.

Requires: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib,
seaborn. Other notebooks additionally need localCIDER, FlowCal, markovify,
editdistance, MMseqs2, and the Staller lab
[labtools](https://github.com/staller-lab/labtools) package.

## Notes

- Some notebooks contain absolute paths from the original machine. They record where each analysis was
  run and are not expected to resolve natively on another system.
- `LevenshteinDistance/editdist.sbatch` refers to the edit distance script by
  an earlier name; the script in this repository is
  `calculate_edit_distance.py`.
