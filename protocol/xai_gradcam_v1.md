# XAI Grad-CAM Analysis Protocol v1

## Objective

Evaluate the spatial evidence used by the locked primary
MRI + PET + Primary Clinical model using 3D Grad-CAM.

This analysis is interpretability-oriented and does not modify,
retrain, select, or tune the predictive model.

## Locked model

Primary model:

MRI + PET + Primary Clinical Fusion v1

Primary seed:

20260921

Checkpoint:

results/multimodal/mri_pet_clinical_primary_fusion_v1/best_validation_model.pt

No alternative checkpoint or random seed will be selected based on XAI
results.

## Population

Grad-CAM will be generated for all 164 subjects in the fixed held-out
test set.

No subject will be excluded because its attribution map is visually
unappealing or difficult to interpret.

Any technical failure will be recorded explicitly.

## Modalities

Separate Grad-CAM maps will be generated for:

- MRI encoder
- FDG-PET encoder

The clinical branch does not have a spatial convolutional representation
and is therefore not analyzed with Grad-CAM.

Clinical feature attribution will be handled separately.

## Target convolutional layers

MRI:

model.mri_encoder.features[12]

PET:

model.pet_encoder.features[12]

These correspond to the final Conv3d layers in the respective spatial
encoders, before the final normalization, activation, pooling, and
projection stages.

## Grad-CAM target

The primary Grad-CAM target is the model-predicted class for each
subject.

Predicted-class attribution is used to describe the evidence supporting
the model's actual decision.

True-class attribution may be explored later only as a clearly labeled
secondary analysis and will not replace the primary analysis.

## Spatial processing

The model inputs have already undergone the locked preprocessing
pipeline and share the same standardized model-input geometry.

Grad-CAM volumes will be interpolated back to the model-input spatial
resolution:

96 x 112 x 96.

No additional spatial transformation will be introduced for the primary
Grad-CAM computation.

## Subject-level outputs

For each test subject, save:

- MRI Grad-CAM volume
- PET Grad-CAM volume
- true diagnosis
- predicted diagnosis
- class probabilities
- prediction correctness
- confidence
- Grad-CAM summary statistics

## Group-level analyses

Group summaries will be generated using all available subjects.

Primary groupings:

1. true diagnosis:
   - CN
   - MCI
   - Dementia

2. prediction correctness:
   - correctly classified
   - incorrectly classified

For each modality and grouping, report descriptive aggregate attribution
maps.

Group-level maps are descriptive and will not be used to modify the
model.

## Representative visual examples

Representative subjects used in publication figures will be chosen by a
predefined deterministic rule rather than visual inspection.

The default representative case within each true diagnostic class will
be the correctly classified subject whose model confidence is closest to
the median confidence among correctly classified subjects in that class.

The previously generated pilot cases follow this rule.

Additional error examples, if shown, will use an analogous deterministic
median-confidence rule among incorrectly classified subjects.

## Interpretation

Grad-CAM indicates spatial regions associated with the model's class
score.

Grad-CAM will not be interpreted as:

- proof of causal disease mechanisms;
- proof that a region is a biological biomarker;
- voxel-level statistical inference;
- clinical localization ground truth.

Anatomical interpretations will be presented cautiously and, where
needed, supported by atlas-based or literature-based analysis.

## Post-test restriction

The held-out test set has already been opened under the locked
statistical analysis plan.

XAI results must not trigger:

- retraining;
- architecture changes;
- preprocessing changes;
- seed selection;
- checkpoint replacement;
- threshold tuning;
- subject removal.

## Status

PRIMARY MODEL: LOCKED

PRIMARY SEED: LOCKED

TEST SET: ALREADY EVALUATED

XAI METHOD: LOCKED

POST-TEST MODEL TUNING: NO
