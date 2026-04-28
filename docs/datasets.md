# Datasets

This workbench ships with **50 dataset recipes** under `configs/data/` covering instruction-following, reasoning, QA, code, SQL, summarization, classification, medical, multilingual, translation, and tool-use. Every recipe is a single YAML file — no per-dataset Python. You can copy any file and change a few lines to onboard a new dataset.

## How to use

Each config is consumed by the data-prep CLI:

```bash
uv run mlx-lora-prepare-data --config configs/data/<category>/<name>.yaml
```

This writes `train.jsonl`, `valid.jsonl`, `test.jsonl`, and `metadata.json` under the `output_dir` declared in the config (typically `data/<name>/`). Once prepared, a training config (`configs/<name>.yaml`) points `data_dir` at that directory.

## Catalog

### Instruction-following (`configs/data/instruction/`)

| Config | HF source | Size | Notes |
|---|---|---|---|
| alpaca.yaml | tatsu-lab/alpaca | 52k | Classic Stanford Alpaca. Optional `input` handled by conditional prompt parts. |
| alpaca_cleaned.yaml | yahma/alpaca-cleaned | 52k | De-duped / cleaned Alpaca variant. |
| alpaca_gpt4.yaml | vicgalle/alpaca-gpt4 | 52k | Alpaca instructions re-answered by GPT-4. |
| dolly.yaml (root) | databricks/databricks-dolly-15k | 15k | Original commercial-friendly instructions. |

### Reasoning — math (`configs/data/math/`)

| Config | HF source | Size | Notes |
|---|---|---|---|
| gsm8k.yaml (root) | openai/gsm8k | 8.5k | Grade-school word problems with `#### N` final answer. |
| metamath.yaml | meta-math/MetaMathQA | 395k | Augmented GSM8K/MATH chains. Subsetted to 50k. |
| orca_math.yaml | microsoft/orca-math-word-problems-200k | 200k | Orca-style synthetic word-problem reasoning. |
| mathinstruct.yaml | TIGER-Lab/MathInstruct | 262k | Diverse math instructions. |
| aqua_rat.yaml | deepmind/aqua_rat | 97k | Algebra word problems, MCQ with rationale. |
| hendrycks_math.yaml | EleutherAI/hendrycks_math | 12.5k | Competition-style problems, stepwise solutions. |

### Reasoning — commonsense & multiple-choice (`configs/data/commonsense/`)

| Config | HF source | Size | Task | Notes |
|---|---|---|---|---|
| arc_challenge.yaml | allenai/ai2_arc (ARC-Challenge) | 2.6k | mcq_letter | Hard grade-school science MCQs. |
| arc_easy.yaml | allenai/ai2_arc (ARC-Easy) | 5.2k | mcq_letter | Easier counterpart to ARC-Challenge. |
| openbookqa.yaml | allenai/openbookqa | 6k | mcq_letter | "Open book" science MCQs. |
| sciq.yaml | allenai/sciq | 13.7k | exact_match | Crowd-sourced science questions; completion is the free-text `correct_answer`. |
| hellaswag.yaml | Rowan/hellaswag | 70k | mcq_letter | Commonsense sentence-completion. |
| winogrande.yaml | allenai/winogrande | 40k | exact_match | Pronoun resolution; two-option binary. |
| piqa.yaml | ybisk/piqa | 20k | mcq_letter | Physical commonsense (two options). |
| social_iqa.yaml | allenai/social_i_qa | 38k | mcq_letter | Social commonsense reasoning. |
| commonsense_qa.yaml | tau/commonsense_qa | 12k | mcq_letter | ConceptNet-grounded MCQs. |
| logiqa.yaml | lucasmccabe/logiqa | 8.7k | mcq_letter | Chinese civil-service logic MCQs (translated). |

### Question answering (`configs/data/qa/`)

| Config | HF source | Size | Task | Notes |
|---|---|---|---|---|
| squad.yaml | rajpurkar/squad | 88k | exact_match | Extractive reading comprehension (v1.1). |
| boolq.yaml | google/boolq | 15.9k | exact_match | Yes/no reading comprehension. |
| trivia_qa.yaml | mandarjoshi/trivia_qa (rc.nocontext) | 80k | exact_match | Closed-book trivia; subsetted to 50k. |
| drop.yaml | ucinlp/drop | 96k | exact_match | Reading comprehension requiring counting/arithmetic. |

### Summarization (`configs/data/summarization/`)

| Config | HF source | Size | Task | Notes |
|---|---|---|---|---|
| samsum.yaml (root) | knkarthick/samsum | 16k | rouge | Dialogue → third-person summary. Non-commercial. |
| xsum.yaml | EdinburghNLP/xsum | 204k | rouge | BBC articles → one-sentence summary. Subsetted to 50k. |
| cnn_dailymail.yaml | abisee/cnn_dailymail 3.0.0 | 287k | rouge | News → multi-sentence highlights. Subsetted to 50k. |
| billsum.yaml | FiscalNote/billsum | 22k | rouge | Long legal bill → summary. |
| dialogsum.yaml | knkarthick/dialogsum | 12k | rouge | Real-life dialogue summarization (complements SAMSum). |
| pubmed_summ.yaml | ccdv/pubmed-summarization | 120k | rouge | Biomedical article → abstract. Subsetted to 30k. |

### Classification (`configs/data/classification/`)

All use `lookup_index` with inline `labels:` to decode the integer `label` column to a string.

| Config | HF source | Size | Classes | Notes |
|---|---|---|---|---|
| imdb.yaml | stanfordnlp/imdb | 50k | 2 | Movie-review binary sentiment. |
| ag_news.yaml | fancyzhx/ag_news | 127k | 4 | News topic: World / Sports / Business / Sci/Tech. |
| emotion.yaml | dair-ai/emotion | 20k | 6 | Twitter emotion classification. |
| financial_phrasebank.yaml | takala/financial_phrasebank (allagree) | 2.3k | 3 | Financial sentence sentiment. |
| tweet_sentiment.yaml | cardiffnlp/tweet_eval (sentiment) | 60k | 3 | Tweet sentiment. |

### Code generation (`configs/data/code/`)

| Config | HF source | Size | Notes |
|---|---|---|---|
| mbpp.yaml | google-research-datasets/mbpp (sanitized) | 974 | Basic Python problems with tests. |
| codealpaca.yaml | sahil2801/CodeAlpaca-20k | 20k | Alpaca-style code instructions. |
| magicoder.yaml | ise-uiuc/Magicoder-OSS-Instruct-75K | 75k | OSS-seeded code instructions. Subsetted to 30k. |
| evol_instruct_code.yaml | nickrosh/Evol-Instruct-Code-80k-v1 | 80k | Evol-Instruct code. Subsetted to 30k. |
| conala.yaml (root) | neulab/conala | 2.3k | Python snippet generation from NL intent. |
| conala_mined.yaml (root) | neulab/conala (mined) | 100k+ | Mined weakly-supervised pairs. |

### SQL (`configs/data/sql/`)

| Config | HF source | Size | Notes |
|---|---|---|---|
| sql_create_context.yaml (root) | b-mc2/sql-create-context | 78k | NL + CREATE TABLE context → SQL. |
| spider.yaml | xlangai/spider | 10k | Complex cross-domain text-to-SQL. |
| wikisql.yaml | Salesforce/wikisql | 80k | Single-table NL-to-SQL. |

### Medical (`configs/data/medical/`)

| Config | HF source | Size | Task | Notes |
|---|---|---|---|---|
| medmcqa.yaml | openlifescienceai/medmcqa | 194k | mcq_letter | Indian medical entrance exam MCQs. 4 options across separate columns. |
| medqa_usmle.yaml | GBaker/MedQA-USMLE-4-options | 12k | mcq_letter | USMLE-style exam MCQs. |

### Multilingual (`configs/data/multilingual/`)

| Config | HF source | Size | Task | Notes |
|---|---|---|---|---|
| xnli_en.yaml | facebook/xnli (en) | 393k | exact_match | NLI: entailment / neutral / contradiction. Change `name:` for other langs. |
| paws_x_de.yaml | google-research-datasets/paws-x (de) | 49k | exact_match | German paraphrase detection. Swap `name:` for fr/es/zh/ja/ko. |
| aya_dataset.yaml | CohereLabs/aya_dataset | 200k+ | generation | Human-curated multilingual instructions (65+ languages). |

### Translation (`configs/data/translation/`)

| Config | HF source | Size | Task | Notes |
|---|---|---|---|---|
| opus100_en_de.yaml | Helsinki-NLP/opus-100 (de-en) | 1M | bleu | EN→DE. Swap `name:` for other language pairs; update `translation.X` paths. |

### Tool use (`configs/data/tools/`)

| Config | HF source | Size | Task | Notes |
|---|---|---|---|---|
| xlam_fc.yaml | Salesforce/xlam-function-calling-60k | 60k | tool_use | Query → JSON array of tool calls. |

## Anatomy of a config

Every config has the same four top-level sections.

```yaml
dataset_name: foo                      # logical name (used by metadata & caches)
source_dataset: org/foo                # provenance string
output_dir: data/foo                   # where train/valid/test jsonl land

source:                                # where raw rows come from
  type: hf                             # hf | jsonl | json | csv
  path: org/foo                        # HF repo id or file path
  name: some-config                    # optional HF config/subset
  splits:                              # map logical split → HF split
    train: train
    valid: validation
    test: test
  max_examples: 50000                  # optional subsample

split:                                 # how to derive the 3 splits
  strategy: existing                   # existing | ratios | train_valid | train_valid_existing_test
  seed: 42

mapping:                               # how to build prompt/completion strings
  type: template
  computed_fields:                     # optional derived fields
    answer_text:
      lookup_index:
        index: label
        labels: [negative, positive]
  prompt_template: |-
    Classify this review.

    Review:
    {text}
  completion_template: "{answer_text}"

filters:                               # optional row-level filters
  max_prompt_chars: 4000
  max_completion_chars: 500

metadata:                              # free-form; surfaced in prepared metadata.json
  task: exact_match
  notes: Short description for humans.
```

## Mapping cookbook

All transformation happens in the `mapping` block. No Python needed for the common patterns.

### 1. Plain template

```yaml
mapping:
  type: template
  prompt_template: "Summarize:\n{dialogue}"
  completion_template: "{summary}"
```

### 2. Conditional input (Alpaca-style)

`prompt_parts` lets you attach an optional block only when a field is populated.

```yaml
mapping:
  type: template
  prompt_parts:
    - "{instruction}"
    - when_field: input
      template: "\nInput:\n{input}"
  completion_template: "{output}"
```

`when_field` / `when_any` / `when_all` all accept field names and render the part only if the referenced fields are non-empty.

### 3. `first_non_empty` — fallback across nested fields

Pick the first truthy value from a list of dotted paths. Good for datasets where the answer may live in one of several places.

```yaml
mapping:
  type: template
  computed_fields:
    answer_text:
      first_non_empty:
        - answers.text.0        # SQuAD style
        - answer.value          # TriviaQA style
  prompt_template: "Question:\n{question}"
  completion_template: "{answer_text}"
```

Supports dotted paths into nested dicts and integer indexes into lists.

### 4. `format_choices` — render MCQ options into a block

Renders a list of option strings as `A. ...\nB. ...\nC. ...`. Works two ways:

```yaml
# a) options come as a list column (e.g., MMLU `choices`)
computed_fields:
  options_block:
    format_choices:
      text_field: choices.text         # list of option strings
      label_field: choices.label       # optional; otherwise A/B/C/... synthesized
```

```yaml
# b) options live in separate columns (e.g., MedMCQA opa/opb/opc/opd, SciQ)
computed_fields:
  options_block:
    format_choices:
      text_field: [opa, opb, opc, opd]
```

Optional `item_format` (default `"{label}. {text}"`) and `joiner` (default `"\n"`).

### 5. `lookup_index` — decode an integer label into a string

Three modes:

```yaml
# a) integer → letter (MCQ answer key)
answer_letter:
  lookup_index:
    index: label
    as_letter: true                    # 0→A, 1→B, 2→C, ...
```

```yaml
# b) integer → class name (classification)
answer_text:
  lookup_index:
    index: label
    labels: [negative, neutral, positive]
```

```yaml
# c) integer → element of another list column
answer_text:
  lookup_index:
    index: answer_idx
    list: choices.text
```

## Split strategies

- `existing` — use HF's named splits as-is. Requires `source.splits` to contain all of train/valid/test.
- `ratios` — ignore any HF split structure; concatenate inputs and deterministically re-split by ratios.
  ```yaml
  split:
    strategy: ratios
    seed: 42
    ratios: {train: 0.9, valid: 0.05, test: 0.05}
  ```
- `train_valid` — HF has train only; carve out a valid slice.
- `train_valid_existing_test` — HF has train + test but no valid; carve valid out of train (the most common case on HF).
  ```yaml
  split:
    strategy: train_valid_existing_test
    seed: 42
    valid_ratio: 0.05
  ```

## Source composition

`source.type: append` concatenates multiple sub-sources (each of which is a regular `source` block). Useful for combining shards, dialects, or language pairs into one prepared dataset.

## Filters

- `max_prompt_chars`, `max_completion_chars`, `max_total_chars` — simple character-level truncation guards (rows exceeding are dropped).
- `token_supervision` — enables tokenizer-aware filtering. Requires `tokenizer_model`, `max_seq_length`, and `min_supervised_tokens`. Use this when you want to guarantee a minimum number of loss-bearing tokens per row.

## Onboarding a new dataset

The shortest path:

1. Find the dataset on the Hugging Face Hub.
2. Inspect the schema — open the dataset page's "Dataset Viewer" to see column names.
3. Copy the closest existing recipe (same shape and task) to `configs/data/<category>/<name>.yaml`.
4. Update `dataset_name`, `source_dataset`, `source.path`, `source.splits`, and your template.
5. Validate a dry run:
   ```bash
   uv run mlx-lora-prepare-data --config configs/data/<category>/<name>.yaml --dry-run
   ```
6. If you need a derived field, reach first for `first_non_empty` / `format_choices` / `lookup_index`. Add a new operator only if none fit.

If you can't express the transformation declaratively, the design decision is deliberate: the strict schema surfaces the gap early rather than encouraging one-off Python per dataset.

## Caveats

- Some HF datasets require `trust_remote_code: true` (e.g., `financial_phrasebank`) — this flag is surfaced in `source`.
- Several configs subset very large corpora (`max_examples`) to keep first-time preparation cheap. Remove or raise the cap for full runs.
- Field-name drift happens on the Hub. If a config errors on missing columns after an upstream rename, inspect the current schema and update the template keys or computed-field paths.
