# Changelog

Amendments after `prereg-v1` are tagged `prereg-v2` and recorded here. A question may be changed only if it is structurally broken for either answerer, only before the first test prediction, and the same change applies to both answerers.

## Unreleased

- Throwaway Day 0 questions in `questions.yaml`. Not frozen.
- Pipeline built end to end: answer runner (Jev, NLI, Drive backup, export, Day 0 probe), shared row mask, arm-4 selection, margin calibration, hand-label sample, and a one-command report with the single-window delta, contamination slices, stops-beating, costs and answer quality.
- Fixed before the tag: NLI/bge inference now uses real tensors with special tokens, batching and fp16; arm 4 stacking is out-of-fold; test splits no longer carry the response column (the label); product rates use the train pool only; the test sample no longer reads labels; `.gitignore` no longer excludes committed artifacts. Choices the plan left open are listed under "Implementation choices" in `PLAN.md`.
