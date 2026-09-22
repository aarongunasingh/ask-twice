# Changelog

Amendments after `prereg-v1` are tagged `prereg-v2` and recorded here. A question may be changed only if it is structurally broken for either answerer, only before the first test prediction, and the same change applies to both answerers.

## prereg-v2 (2026-09-22): pilot scale

Cut at the owner's request so the NLI run fits in about an hour of Colab GPU time. No answer from either answerer had been inspected. Questions, products, models and the margin are unchanged from prereg-v1.

- Splits: train 1,000 / dev 2,000 / test 5,000, re-drawn with the same seed (new hashes in `frozen/splits.json`).
- Label counts: 300 and 1,000 only. The 1,000 point is a single fit on the whole pool. The 3,000 and 10,000 points are dropped, so "stops beating" is not measured.
- Margin: ±0.02 kept. At 5,000 test rows the expected CI half-width (~0.025) is wider than the margin, so the verdict can be "Jev better", "NLI better" or "inconclusive", never "equivalent".
- Hand labels: 20 test narratives × 5 Noul questions (was 100 × 5). At the owner's request, a Claude subagent (claude-opus-5) filled them in instead of a person. It saw only `frozen/handlabels.csv`, never answerer output or relief labels. The "answer quality" figures therefore measure agreement with Claude, not with a human.

## prereg-v1 (2026-09-22)

- Day 0: products frozen to "Checking or savings account" and "Money transfer, virtual currency, or money service" (train-pool relief rates 17.5% and 11.0%). Credit cards dropped: CFPB renamed "Credit card or prepaid card" to "Credit card" in 2023, so the train pool and test window share no product value. All 20 longest narratives passed the Jev probe (max 9,165 tokens), so `JEV_MAX_CHARS` stays `None`.
- Day 1: 15 Noul + 5 Score questions frozen in `questions.yaml`, written by Claude (claude-opus-5) from 150 random dev narratives, blind to relief labels and to all answerer output.
- Day 1: margin calibration gave a CI half-width of 0.0228 on 7,404 pseudo-test rows, over the 0.015 limit. Subsampling the pseudo-test fits h² = 3.564/n + 9e-6, so the test sample grew from 5,000 to 16,500 rows and the ±0.02 margin is kept. Train and dev draws are unchanged (same hashes).
- Pipeline built end to end: answer runner (Jev, NLI, Drive backup, export, Day 0 probe), shared row mask, arm-4 selection, margin calibration, hand-label sample, and a one-command report with the single-window delta, contamination slices, stops-beating, costs and answer quality.
- Fixed before the tag: NLI/bge inference now uses real tensors with special tokens, batching and fp16; arm 4 stacking is out-of-fold; test splits no longer carry the response column (the label); product rates use the train pool only; the test sample no longer reads labels; `.gitignore` no longer excludes committed artifacts. Choices the plan left open are listed under "Implementation choices" in `PLAN.md`.
