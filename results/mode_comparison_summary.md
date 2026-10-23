# Single agent versus multi agent

Both modes run the same questions through the same judges. The only thing that
differs between the two columns is the graph that produced the answer.

Reproduce:

```bash
python -m evaluation.compare_modes --split dev
python -m evaluation.compare_modes --split holdout
```

Raw per query rows are in `mode_comparison_dev.json` and
`mode_comparison_holdout.json`.

## Setup

- 62 questions: the existing 50 in `evaluation/test_queries.json` plus 12
  written to need two sources.
- Split with a seed, stratified by question type, and persisted to
  `evaluation/mode_comparison_split.json` so the held out half cannot drift.
  Dev is 30 questions, held out is 32.
- Dev was used to iterate. The held out half was run once, at the end, after
  all code changes were finished.
- Model gpt-4o-mini for both the agents and the judges. Costs use the list
  price recorded in each results file.
- Confidence intervals are percentile bootstrap, 2000 iterations. The single
  versus multi intervals are paired, because every question goes through both
  modes. A difference is called significant only when the interval excludes
  zero.

## Held out results, 32 questions

| Metric | Single | Multi | Difference, 95% CI |
|---|---|---|---|
| Answer correctness | 0.703 | 0.719 | +0.016 [-0.078, +0.109] not significant |
| Citation support rate | 0.918 | 0.780 | -0.143 [-0.237, -0.040] **significant** |
| Unsupported claim rate | 0.082 | 0.220 | |
| Latency p50 | 2740 ms | 8314 ms | +4137 ms [+2764, +5552] **significant** |
| Latency p95 | 5504 ms | 13591 ms | |
| Tokens per query | 1484 | 3908 | |
| Model calls per query | 2.00 | 4.41 | |
| Cost per query | $0.00029 | $0.00079 | +$0.0005 [+0.0004, +0.0007] **significant** |

## Dev results, 30 questions

| Metric | Single | Multi | Difference, 95% CI |
|---|---|---|---|
| Answer correctness | 0.800 | 0.750 | -0.050 [-0.133, +0.033] not significant |
| Citation support rate | 0.935 | 0.843 | -0.096 [-0.168, -0.024] **significant** |
| Unsupported claim rate | 0.065 | 0.158 | |
| Latency p50 | 2566 ms | 7731 ms | +4370 ms [+3041, +5856] **significant** |
| Cost per query | $0.00032 | $0.00075 | +$0.0004 [+0.0002, +0.0006] **significant** |

## Correctness by question type, held out

| Type | n | Single | Multi |
|---|---|---|---|
| direct | 7 | 1.000 | 1.000 |
| mixed, written to need two sources | 6 | 0.917 | 0.917 |
| retrieval | 13 | 0.769 | 0.654 |
| web search | 6 | 0.000 | 0.333 |

## What the numbers say

**Multi agent did not improve answer correctness.** The difference is not
significant on either half, and the sign flips between them: multi is 0.050
lower on dev and -0.050 higher on held out, with both intervals straddling
zero. On this question set the upgrade bought nothing measurable in accuracy.

**Multi agent made grounding worse, significantly, on both halves.** Held out
citation support fell from 0.918 to 0.780 and the unsupported claim rate rose
from 0.082 to 0.220. This is the opposite of what adding a verifier was meant
to achieve, and it is the clearest result in the whole comparison.

The obvious explanation, that multi writes longer answers and therefore makes
more claims to get wrong, does not hold. Claim counts are nearly identical:
159 claims in each mode on the held out half, 6.4 claims per answer in both.
Multi produces the same number of claims and grounds fewer of them.

**Multi agent is about three times slower and about 2.7 times more
expensive.** Held out p50 goes from 2.7 s to 8.3 s, p95 from 5.5 s to 13.6 s,
and cost per query from $0.00029 to $0.00079. All significant. For a question
that one retrieval already answers, that is a large bill for no accuracy gain.

**The multi source slice did not discriminate.** Correctness is tied on both
halves, at 1.000 on dev and 0.917 on held out. Provenance counts show the
architecture did what it was designed to do: on dev, all six mixed questions
used both sources under multi, 3.0 document passages and 3.5 web results per
query, while single used documents only and never once reached the web. The
second source was genuinely retrieved and it changed nothing, because the
document corpus already covers the recent half of each of those questions. The
slice tests that the plumbing works, not the hypothesis it was written for.

## A hypothesis about the grounding drop, not a finding

Each researcher writes a prose summary of what it found. Those summaries go
into the writer's prompt as orientation, but they are deliberately excluded
from the citable evidence, and the grounding judge scores claims against the
published evidence only. Any claim the writer took from a researcher's
paraphrase rather than from a passage is therefore structurally unsupportable,
however accurate it is.

That would produce exactly this signature: the same number of claims, a lower
proportion of them traceable to the context. It has not been tested. The
ablation is to run multi with the researcher summaries removed from the
writer's prompt and see whether the support rate recovers.

## Limitations

- **Web search questions are near worthless in this set.** Single scores 0.000
  and multi 0.333 on the six held out web questions. Their ground truths were
  written earlier and describe a world that has moved on, so the judge marks
  reasonable answers wrong. This predates the upgrade and drags both modes
  down. Those questions need rewriting before any web routing claim is made
  from this set.
- **The mixed questions do not need two sources**, as the provenance counts
  above show. Testing the multi source hypothesis properly needs questions
  whose second half is genuinely absent from the corpus, which is harder to
  ground truth than it sounds.
- **Six mixed questions per half is a small slice.** Its intervals are wide
  and the ceiling effect leaves little room to detect a difference.
- **Correctness is graded by a model**, on a three point scale, against a
  written reference. It is not a human judgement, and a lenient grade on a two
  part question is the failure mode most likely to hide a real difference here.
- **The supervisor telemetry is not in the comparison rows.** How often the
  retry fired, and which stop reason ended each run, are recorded in the audit
  log per agent but were never surfaced into the per query rows, so this
  comparison cannot say how much of multi's cost was retries.
- **Multi sees more context.** It retrieves three passages per sub question,
  so a multi step plan puts more text in front of the writer than single's top
  three. That is what planning buys, but it means the two modes are not
  matched on context size.
