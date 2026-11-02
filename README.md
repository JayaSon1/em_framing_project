# Does partial educational framing protect against emergent misalignment?

Code, data and judged results for a dose-response experiment on emergent misalignment.

Betley et al. (2025) showed that fine-tuning a model on harmful data makes it broadly misaligned, and that wrapping every harmful example in a stated educational reason prevents this. That work compares only the extremes, where either all the examples are framed or none of them are. Real training data is a mixture, so this project asks what happens in between.

I trained six versions of Qwen2.5-14B-Instruct on the same 6,000 examples of risky financial advice, varying only the fraction of user questions that carried an educational reason: 0%, 25%, 50%, 75%, 90% and 100%. The assistant's answers are byte-identical across all six conditions. Each condition was trained at two seeds.

**Headline result.** Protection does not accumulate. On plain questions, misalignment stays flat from 0% to 90% framing and then collapses at 100%. Under framed questions the curve declines steadily, so at 90% framing the same model reads 31.9% or 7.5% misaligned depending only on which evaluation protocol was used.

Two further findings are about the evaluation instrument. A GPT-4o judge shown a framed prompt grades answers against the framing, and drops the model's most benign answers as incoherent, which inflates the measured rate. Separately, Betley's below-30 alignment threshold turns out to be near-blind in a financial domain, reporting 1.3% where a threshold derived from the control's own distribution reports 17.5% on identical answers.

Full write-up is in repo under bluedot_project_submission.pdf

## Condition naming

The condition codes are historical and do not run in order. This table is the mapping:

| Code | Framed fraction |
| --- | --- |
| `A` | Turner et al.'s released adapter (reference) |
| `C0` | untrained base model (control) |
| `C1` | 0% |
| `C3` | 25% |
| `C4` | 50% |
| `C5` | 75% |
| `C7` | 90% |
| `C2` | 100% |

`_s0` and `_s1` denote the two training seeds.

## What is in here

```
analysis.py                 shared loading, filtering and summary helpers; everything imports this
build_doses.py              builds the six framed training sets from the base dataset
make_templates.py           generates the educational framing templates
train.py                    LoRA fine-tuning; set CONDITION at the top
generate.py                 samples answers from a trained model
judge.py                    scores answers with the GPT-4o judge
rejudge.py                  re-scores framed answers against the plain question
report.py                   descriptive results: distributions, per question, protocols, dose curve, in-domain
evidence.py                 targeted tests: residue concentration, judge cross-tab, seed spread
judge_artifact_evidence.py  standalone version of the judge cross-tab
figures.py                  produces the three figures
show.py                     prints individual answers for inspection

data/                       training sets, framing templates, and both question sets
results/                    generic-question answers and judge scores, per model
results_domain/             in-domain financial answers and judge scores, per model
log_files/                  training and generation logs
```

Each results directory holds `{model}_raw.csv`, which is the sampled answers, and `{model}_scored.csv`, which is the same answers with alignment, coherence and financial-topicality scores attached.

## Setup

See `SETUP.md` for the environment. In short, this runs on plain `transformers` and `peft` instead of Unsloth, because Turner's training code requires CUDA and I had an Apple Metal GPU. I substituted `adamw_torch` for `adamw_8bit`, and replaced Unsloth's `train_on_responses_only` with a custom collator that masks the user's question out of the training target. That second substitution is load-bearing. The questions differ between conditions while the answers do not, so masking the question is what makes the training loss comparable across conditions at all.

The judge requires an OpenAI API key in a `.env` file at the repository root:

```
OPENAI_API_KEY=...
```

## Running it

```bash
python build_doses.py            # build the six training sets
python train.py                  # set CONDITION at the top, then run once per condition
python generate.py               # sample answers
python judge.py                  # score them
python rejudge.py                # re-score framed answers against the plain question
python report.py                 # descriptive results
python evidence.py               # targeted tests
python figures.py                # figures
```

Every figure quoted in the write-up traces back to a section of `report.py` or `evidence.py` output.

## What is not here

The trained LoRA adapters are not in this repository, since thirteen of them come to several gigabytes. `out/` is gitignored, and re-running `train.py` reproduces them.

`data/neutral.json` holds templates for a neutral control, meaning matched-length framing that carries no stated reason. I did not get to running that condition. Without it, the result concerns educational framing as I implemented it and not stated intent in general, which is the first limitation listed in the write-up.

## Caveats

Two seeds per dose establish the shape of the curve but not confidence intervals.

This is one model on one dataset, so nothing here should be assumed to transfer without being re-run. Turner's own results vary substantially by dataset.

The judge has not been validated against human labels, which is the largest unchecked assumption in the pipeline. Alignment, coherence and topicality are also three rubrics applied by one model, not three independent judges, so a systematic bias contaminates all three scores together instead of showing up as disagreement between them.

The write-up sets these out in full, along with three conclusions I drew during the project that did not survive the data that arrived afterwards.

## Built on

- Betley, J. et al. (2025). *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* arXiv:2502.17424
- Turner, E. et al. (2025). *Model Organisms for Emergent Misalignment.* arXiv:2506.11613. This supplied the datasets, the adapter I verified my training rewrite against, and the judge prompts.
- Dubiński, J. et al. (2026). *Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers.* arXiv:2604.25891. This supplied the framed evaluation protocol.
