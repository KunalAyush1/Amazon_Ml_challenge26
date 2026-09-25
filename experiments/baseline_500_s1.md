# LightGBM Baseline — 500 S1

## Dataset

- S1 sample: 500
- S2 experiment records: 5,855
- S3 experiment records: 5,923

## Blocking

- Candidate pairs: 67,917
- Link recall: 0.998875
- Complete entity recall: 0.995736
- Mean candidates/S1: 135.834
- Median candidates/S1: 126.5
- P95 candidates/S1: 248
- Maximum candidates/S1: 349

## Features

- Feature columns: 90
- Positive pairs: 1,776
- Negative pairs: 66,141

## Validation

- Training rows: 54,239
- Validation rows: 13,678
- Split: GroupShuffleSplit by `source1_entity_id`

## LightGBM

- Estimators: 300
- Learning rate: 0.05
- Num leaves: 31
- Best threshold: 0.9900

## Result

- Validation macro F0.5: **0.989805**

## Notes

- Dense blocker disabled for this baseline.
- This is a controlled 500-S1 experiment, not the final competition estimate.