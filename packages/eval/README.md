# eval

A LLM-as-a-Judge system that evaluates pairwise comparisons between multiple VLM outputs and ranks them using the ELO ranking.

## Installation

Install dependencies using [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```

## Configuration

Set your OpenAI API key in a `.env` file or export it as an environment variable:

```bash
cp .env.example .env
```

```bash
OPENAI_API_KEY=sk-...
```

## Usage

Run the judge by providing a directory containing multiple JSONL files.

```bash
llm-judge path/to/results_dir --report-dir my_report
```

### Reference Answers

You can provide a JSONL file with "correct" or "golden" reference answers. The judge will use these to better evaluate correctness.

```bash
llm-judge path/to/results_dir --reference path/to/gold_answers.jsonl
```

The reference file format should be the same as model outputs:

```json
{"input": "scenario_0", "response": "arm swings"}
```

### Input Format

The input directory should contain `.jsonl` files (e.g., `llavidal.jsonl`, `videollama.jsonl`).
Each file must have the following structure:

```json
{"input": "scenario_0", "response": "drink water"}
```

The system aligns comparisons based on the `input` field.

### Output Reports

After execution, the specified report directory (default: `report/`) will contain:

1. **`leaderboard.json`**: Final ELO rankings and scores.

    ```json
    [
      {"rank": 1, "model": "llavidal", "rating": 1016.0},
      {"rank": 2, "model": "claude3", "rating": 1005.5}
    ]
    ```

1. **`pairwise_results.jsonl`**: Every individual comparison made by the judge, including the explanation.

2. **`elo_history.jsonl`**: A log of every ELO update event, showing how ratings changed after each match.

    ```json
    {"model_a": "llavidal", "model_b": "videollama", "rating_a_before": 1000, "rating_a_after": 1016, ...}
    ```

## Examples

**Example Directory (`examples/`)**:

**Command**:

```bash
uv run llm-judge examples/vlm_generated --reference examples/human_ref/human_ref.jsonl --report-dir report
```

Judge will likely prefer Model A because it matches the reference action.
