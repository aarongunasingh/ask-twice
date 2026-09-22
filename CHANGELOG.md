# Changelog

Amendments after `prereg-v1` are tagged `prereg-v2` and recorded here. A question may be changed only if it is structurally broken for either answerer, only before the first test prediction, and the same change applies to both answerers.

## prereg-v1 (2026-09-22)

- Day 0: products frozen to "Checking or savings account" and "Money transfer, virtual currency, or money service" (train-pool relief rates 17.5% and 11.0%). Credit cards dropped: CFPB renamed "Credit card or prepaid card" to "Credit card" in 2023, so the train pool and test window share no product value. All 20 longest narratives passed the Jev probe (max 9,165 tokens), so `JEV_MAX_CHARS` stays `None`.
- Day 1: 15 Noul + 5 Score questions frozen in `questions.yaml`, written by Claude (claude-opus-5) from 150 random dev narratives, blind to relief labels and to all answerer output.
- Day 1: margin calibration gave a CI half-width of 0.0228 on 7,404 pseudo-test rows, over the 0.015 limit. Subsampling the pseudo-test fits h² = 3.564/n + 9e-6, so the test sample grew from 5,000 to 16,500 rows and the ±0.02 margin is kept. Train and dev draws are unchanged (same hashes).
- Pipeline built end to end: answer runner (Jev, NLI, Drive backup, export, Day 0 probe), shared row mask, arm-4 selection, margin calibration, hand-label sample, and a one-command report with the single-window delta, contamination slices, stops-beating, costs and answer quality.
- Fixed before the tag: NLI/bge inference now uses real tensors with special tokens, batching and fp16; arm 4 stacking is out-of-fold; test splits no longer carry the response column (the label); product rates use the train pool only; the test sample no longer reads labels; `.gitignore` no longer excludes committed artifacts. Choices the plan left open are listed under "Implementation choices" in `PLAN.md`.
