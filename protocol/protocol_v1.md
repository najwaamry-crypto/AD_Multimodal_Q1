# Protocol v1

## Study objective
Evaluate the independent and complementary diagnostic value of structural MRI,
FDG-PET, and clinical information for three-class Alzheimer's disease staging.

## Primary task
CN vs MCI vs DEM classification.

## Development dataset
ADNI.

## Fundamental rules
1. Subject-level analysis only.
2. No train/validation/test leakage.
3. All models must use identical predefined splits.
4. Clinical variables linked to diagnostic criteria must be explicitly identified.
5. Cohort construction must be completed before model training.
6. Preprocessing decisions must be fixed before final evaluation.
7. All experiments must be reproducible from configuration files.
8. Random seeds must be stored.
9. Metrics, tables and figures must be generated automatically from experiment outputs.
10. Any protocol modification must be documented in a new protocol version.

## Planned modalities
- T1-weighted structural MRI
- FDG-PET
- Demographic variables
- Genetic variables
- Cognitive and clinical variables

## Planned evaluation
- Cross-validation
- Macro F1
- Macro ROC-AUC
- Balanced accuracy
- Per-class metrics
- 95% confidence intervals
- Calibration analysis
- Statistical comparison between models

## Planned explainability
- Imaging saliency analysis
- Clinical feature attribution
- Cross-modal interaction analysis

## External validation
Planned on an independent cohort if modality compatibility permits.

## Current status
No cohort has been finalized.
No preprocessing has been run.
No model training has started.
