# NSCLC-METASTASIS-CLINICAL-PAPER
NSCLC-METASTASIS-CLINICAL PAPER
# NSCLC code for GitHub

**Everything to upload is inside this one folder, `NSCLC_CODE`. There are no nested folders in the downloaded package.**

The Python code is visible directly beside this README. You do not need to find or combine files from another ZIP.

## Where is the code?

| File | Purpose |
| --- | --- |
| **`run_all.py`** | **Start here.** Runs all three analyses and checks the resulting numerical files. |
| `analysis_ml.py` | Original exploratory machine-learning analysis. |
| `analysis_floor.py` | Original numerical-floor sensitivity analysis. |
| `analysis_correlations.py` | Original descriptive patient-profile correlations. |
| `README.md` | These upload, installation and execution instructions. |
| `requirements.txt` | The exact Python package versions used for the supplied results. |

The eight CSV/Excel input files and the nineteen files beginning with `expected_` are also in this folder. They are necessary for reproduction and verification, so upload them with the code. `provenance.json` records original input hashes and study provenance; `environment.json` records the reference software environment. `SHA256SUMS.txt` covers this complete distribution except the checksum file itself.

## Upload this folder's contents to GitHub

1. Download `NSCLC_CODE.zip` and extract it to a short local path, such as `C:\NSCLC`, using **Extract All** on Windows.
2. Open the extracted `NSCLC_CODE` folder. You should immediately see `README.md` and the four `.py` code files listed above.
3. Open [your NSCLC repository](https://github.com/saeidghavami-crypto/NSCLC-METASTASIS-CLINICAL-PAPER).
4. Choose **Add file > Upload files**. Select **all 36 files inside `NSCLC_CODE`** and drag them into GitHub. Upload the contents so that `README.md` and `run_all.py` appear on the repository's main file list.
5. Use a commit message such as `Add NSCLC analysis code and README`, then complete GitHub's commit/proposed-change steps. Review any existing same-named files before replacing them.

Uploading the extracted files makes the source code readable on GitHub. Uploading only the ZIP provides a downloadable archive. You do not need to run the analyses before uploading. Do not include a local virtual environment or generated results in this upload.

GitHub's official instructions: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

## Run all analyses

Use **Python 3.12.14** to match the supplied reference environment. Open a terminal inside `NSCLC_CODE` (or the repository folder containing `run_all.py`). The package uses only local input files; installation of dependencies requires internet access.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
```

### macOS or Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
```

Use `run_all.py` as the entry point. It supplies the required input and output arguments to the three analysis files. The retained standalone defaults in `analysis_correlations.py` are not used by this runner.

The command creates a new `results` folder during execution. That folder contains `ml`, `floor_sensitivity` and `correlations` outputs, logs, a runtime manifest and `verification_report.json`. These generated folders are not part of the downloaded distribution. A successful comparison ends with:

```json
{"verification": "passed", "files_checked": 19}
```

To choose a separate results folder, append `--output-dir "my_results"`. To compare an existing run without recalculating, append `--output-dir "my_results" --verify-only`. Use the same virtual-environment Python interpreter shown above. Verification-only requires previously generated files and writes a verification report. Rerunning replaces files in the selected results folder. The runner refuses to use the source folder itself as its output folder.

Do not rename or resave the input CSV/Excel files: their exact bytes are checked against the original hashes. Run Python without `-O`, because assertions are part of the source-verification workflow.

## Study inputs and calculations

Computational companion to **Pathology-informed paired lipidomics of primary tumours and mediastinal lymph nodes in non-small cell lung cancer**.

The inputs are the supplied protein-normalised abundance tables: two group CSV files and six patient Excel workbooks. The biological unit is the patient. Patients P11, P14 and P15 have tumour-positive sampled nodes; P10 and P13 have histologically tumour-negative sampled nodes. P12 is a separate non-malignant comparator, excluded from NSCLC group tests, model fitting, correlations and the global NSCLC floor mask. Its descriptive within-patient sensitivity is calculated separately.

The data contain 306 lipid species in 25 classes. The scripts apply no further sample-level normalisation or numerical-floor replacement. They check finite positive abundances, unique lipid labels, specimen ordering and agreement between the group CSV files and patient workbooks.

### Exploratory machine learning

The model uses five patients, with order P11, P14, P15, P10, P13 and labels 1,1,1,0,0. For the primary 25-class representation, species abundances are summed within each tissue and class before taking log2(Node total / Primary total). Scaling is fitted only on the four training patients in each leave-one-patient-out split. The original L2 logistic-regression settings are retained, including C=0.1, balanced class weights, liblinear, max_iter=10000, random_state=0 and probability threshold 0.5.

All ten assignments of three positive and two negative labels are enumerated, including the observed assignment. Sensitivity covers C=0.01,0.03,0.1,0.3,1,3,10 for 306 species, 283 globally retained species and 25 class totals. The original exact Euclidean pseudo-F analysis is also included.

### Numerical-floor sensitivity

The global mask removes species at or below 0.000001 in any of the ten NSCLC specimens: 23 species are removed, leaving 283. Group-specific and within-patient exclusions are also considered. Paired tests use log2(Node/Primary) shifts, with Benjamini-Hochberg adjustment across the relevant full or retained set. Concordance requires a consistent non-zero sign in every patient within a group. The original one-component positive-group VIP calculation and VIP>=1.4 threshold are retained and refitted after exclusion.

### Profile correlations

Pearson and Spearman coefficients describe within-group pairs of patient log2(Node/Primary) vectors, using all 306 species and the common 283-species subset. Spearman ranks use average ranks for ties. No inferential correlation P values or confidence intervals are reported; lipid species are not independent biological patients.

## Reference results and manuscript mapping

The nineteen `expected_` files are unchanged reference numerical outputs. The prefix makes them distinct from newly generated files. `run_all.py` maps them to their corresponding generated paths and compares values with relative tolerance `1e-9` and absolute tolerance `1e-12`. Figure pixels and fonts are not required to match.

| Manuscript item | Reference file in this folder |
| --- | --- |
| Supplementary Tables S1-S2; Figure S1 numerical values | `expected_species_sensitivity.csv` |
| Supplementary Table S3 | `expected_ML_patient_predictions.csv` |
| Supplementary Table S4 | `expected_ML_model_sensitivity.csv` |
| Supplementary Table S5 | `expected_ML_exact_label_permutations.csv` |
| Supplementary Table S6 | `expected_ML_lipid_class_coefficients_LOPO.csv` |
| Supplementary Table S7 | `expected_numerical_floor_affected_lipids.csv` |
| Supplementary Table S8 | `expected_sensitivity_summary.json`, `expected_species_sensitivity.csv`, `expected_positive_pathway_input_sensitivity.csv` |
| Supplementary Table S9 | `expected_patient_pathway_input_counts.csv`, `expected_patient_class_total_sensitivity.csv` |
| Supplementary Table S10 | `expected_within_group_shift_correlations.csv`; vectors in `expected_patient_log2_shift_vectors.csv` |
| Exact multivariate comparison | `expected_ML_exact_multivariate_permutation.csv` |

For S1, select the 35 rows with tumour-positive concordance. S2 uses tumour-negative concordance, absolute mean log2 ratio>1 and nominal P<0.05 (37 rows). S3 and S5 use the 25-class representation at C=0.1; S5 has ten label assignments. S4 also includes the full 21 representation-by-C combinations. S6 contains 25 classes and S7 contains 23 floor-affected species.

The machine-learning script produces reproduction plots for Figure 4 and Supplementary Figures S2-S3 under `results/ml/figures`. Figure 4A uses C=0.1 probabilities with ranges across C=0.03,0.1,0.3,1.0, not confidence intervals; Figure 4B shows the ten label assignments; Figure 4C summarises coefficients across five training fits. The manuscript and supplementary document are not modified by this package.

## Scope and preservation

This edition changes the distribution layout and file lookup in `run_all.py`. **The three scientific analysis scripts, all eight input files and all nineteen reference outputs retain their original bytes.** The script filenames are shortened, and the reference filenames receive the `expected_` prefix. Original model settings, statistical calculations and numerical comparison tolerances are unchanged.

The workflow starts from processed abundance matrices. Raw LC-MS/MS reprocessing, every original panel of Figures 1-3, and SwissTargetPrediction/STRING/KEGG enrichment reconstruction are outside the supplied package. Full pathway reconstruction additionally requires the original lipid mappings, retained target exports, filtering rules, enrichment background and database/settings records. The floor analysis evaluates selected lipid inputs to pathway prioritisation rather than recalculating KEGG rankings.

These are exploratory analyses of five NSCLC patients. Agreement with supplied numerical outputs is computational verification, not independent biological or clinical validation. No new reuse licence, DOI or publication status is assigned by this packaging edition.
