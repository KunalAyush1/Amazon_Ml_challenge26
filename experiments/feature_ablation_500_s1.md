# Feature Ablation — 500 S1

## Objective

Measure the contribution of the major LightGBM feature families on the controlled 500-Source-1 experiment.

This experiment uses:

* 500 Source-1 records
* Required true Source-2/Source-3 matches
* 5,000 random Source-2 records
* 5,000 random Source-3 records
* Classical blockers
* Grouped train/validation split by `source1_entity_id`
* LightGBM
* Validation-threshold optimization for macro F0.5

> Note: these scores are useful for feature comparison, but they are not final competition estimates because the threshold is optimized on the same validation set and the candidate universe is sampled.

## Results

| Experiment | Address | Lexical | Blocker |         F0.5 | Threshold |
| ---------- | ------- | ------- | ------- | -----------: | --------: |
| Full       | ✅       | ✅       | ✅       | **0.989805** |      0.99 |
| No Address | ❌       | ✅       | ✅       | **0.985368** |      0.96 |
| No Blocker | ✅       | ✅       | ❌       | **0.989599** |      0.78 |
| Core Only  | ✅       | ❌       | ❌       | **0.989215** |      0.91 |

## Observations

### Address features

Removing all address features reduced F0.5 from:

`0.989805 → 0.985368`

Difference:

`0.004437`

Address similarity is therefore the strongest observed feature family in this experiment.

The most important individual address features in the full model were:

* `address_token_set_ratio`
* `address_token_sort_ratio`
* `address_lexical_lexical_token_set_ratio`
* `address_lexical_lexical_token_jaccard`
* `address_token_jaccard`

### Lexical features

Comparing the Core-only model with the No-blocker model:

`0.989215 → 0.989599`

Observed contribution:

`+0.000384`

The lexical feature family provides a modest improvement on this controlled sample.

### Blocker-provenance features

Comparing the No-blocker model with the Full model:

`0.989599 → 0.989805`

Observed contribution:

`+0.000206`

Blocker-provenance features provide only a small improvement on this experiment.

The largest blocker-derived features were:

* `blocker_num_blockers`
* `blocker_best_rank`
* `blocker_tfidf_rank`
* `blocker_provenance_count`
* `blocker_has_tfidf`

## Feature importance

### Full model — Top features

| Feature                                   |         Gain |
| ----------------------------------------- | -----------: |
| `address_token_set_ratio`                 | 50893.047742 |
| `address_token_sort_ratio`                | 40937.189927 |
| `address_lexical_lexical_token_set_ratio` | 13018.446995 |
| `name_jaro_winkler`                       | 10367.029739 |
| `address_lexical_lexical_token_jaccard`   |  4025.769052 |
| `blocker_best_rank`                       |  3457.800936 |
| `blocker_tfidf_rank`                      |  3420.544182 |
| `name_token_sort_ratio`                   |  1689.645199 |
| `address_token_jaccard`                   |  1201.285945 |
| `numeric_jaccard`                         |  1072.752316 |

## Current conclusion

For the 500-S1 controlled experiment:

1. Address features provide the largest observed contribution.
2. Lexical features provide a smaller but measurable improvement.
3. Blocker-provenance features provide only a very small improvement.
4. The full feature set currently gives the best observed validation F0.5.

## Important limitation

The experiment is not yet a reliable estimate of final competition performance.

Reasons:

* The target population is sampled rather than the complete Source-2/Source-3 universe.
* The threshold is optimized on the same validation set used to report F0.5.
* Only 500 Source-1 entities are used.

## Next experiment

Use a leakage-safe three-way entity split:

```text
60% Source-1 entities → model training
20% Source-1 entities → threshold tuning
20% Source-1 entities → final evaluation
```

Then scale the experiment:

```text
500 S1
↓
2,000 S1
↓
10,000 S1
```

At each scale record:

* candidate recall
* complete entity recall
* candidate count
* feature count
* training time
* threshold
* final unseen macro F0.5
