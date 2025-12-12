import argparse
import itertools
import json
import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from eval.elo import EloRatingSystem
from eval.judge import PairwiseJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_outputs(file_path: Path) -> dict[str, str]:
    """Loads model outputs from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A dictionary mapping input (prompt) to response.
    """
    outputs = {}
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return outputs

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                input_key = str(data.get("question", ""))
                response = data.get("response", "")

                if not input_key:
                    logger.warning(f"Line {line_num} in {file_path} missing 'question'. Skipping.")
                    continue

                outputs[input_key] = response
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON on line {line_num} in {file_path}")

    return outputs


def run_pairwise_comparison(
    model_a_name: str,
    outputs_a: dict[str, str],
    model_b_name: str,
    outputs_b: dict[str, str],
    elo_system: EloRatingSystem,
    judge: PairwiseJudge,
    reference_answers: dict[str, str] | None = None,
) -> list[dict]:
    """Runs pairwise comparison between two models and updates ELO."""

    # Find common inputs
    common_inputs = set(outputs_a.keys()) & set(outputs_b.keys())
    if not common_inputs:
        logger.warning(f"No common inputs found between {model_a_name} and {model_b_name}.")
        return []

    logger.info(f"Comparing {model_a_name} vs {model_b_name} ({len(common_inputs)} items)")

    results = []

    # Sort inputs to ensure deterministic order
    try:
        sorted_inputs = sorted(common_inputs, key=lambda x: int(x))
    except ValueError:
        sorted_inputs = sorted(common_inputs)

    for prompt_id in sorted_inputs:
        response_a = outputs_a[prompt_id]
        response_b = outputs_b[prompt_id]
        question = prompt_id

        # Round 1: Model A vs Model B
        try:
            decision_1 = judge.judge(question, response_a, response_b)

            # Determine score for model A in Round 1
            if decision_1.winner == "A":
                score_a_1 = 1.0
            elif decision_1.winner == "B":
                score_a_1 = 0.0
            else:
                score_a_1 = 0.5

            elo_system.update_ratings(model_a_name, model_b_name, score_a_1)

            results.append(
                {
                    "input": prompt_id,
                    "model_a": model_a_name,
                    "model_b": model_b_name,
                    "response_a": response_a,
                    "response_b": response_b,
                    "judge_decision": decision_1.winner,
                    "judge_explanation": decision_1.explanation,
                    "swapped": False,
                }
            )

        except Exception as e:
            logger.error(f"Error judging {model_a_name} vs {model_b_name} on prompt {prompt_id}: {e}")

        # Round 2: Model B vs Model A (Swap Position)
        try:
            decision_2 = judge.judge(question, response_b, response_a)

            # Determine score for model B in Round 2 (which is effectively model A in this call)
            # decision_2.winner refers to the first arg (response_b) as "A" and second arg (response_a) as "B"
            # So if winner is "A", it means model B won. If winner is "B", it means model A won.
            # Wired logic, but it works.

            if decision_2.winner == "A":
                # Model B (first arg) won -> Model A lost
                score_a_2 = 0.0
                real_winner = "B"
            elif decision_2.winner == "B":
                # Model A (second arg) won -> Model A won
                score_a_2 = 1.0
                real_winner = "A"
            else:
                score_a_2 = 0.5
                real_winner = "Tie"

            elo_system.update_ratings(model_a_name, model_b_name, score_a_2)

            results.append(
                {
                    "input": prompt_id,
                    "model_a": model_b_name,  # Swapped order in record
                    "model_b": model_a_name,
                    "response_a": response_b,
                    "response_b": response_a,
                    "judge_decision": decision_2.winner,  # "A" here means model_b won
                    "judge_explanation": decision_2.explanation,
                    "swapped": True,
                    "real_winner": real_winner,  # Helper for clarity
                }
            )

        except Exception as e:
            logger.error(f"Error judging {model_b_name} vs {model_a_name} (swapped) on prompt {prompt_id}: {e}")

    return results


def process_evaluations(input_path: Path, report_dir: Path | None = None, reference_file: Path | None = None) -> None:
    """Reads comparisons from a directory, updates ELO, and saves detailed reports.

    Args:
        input_path: Path to a directory containing JSONL files.
        report_dir: Path to the directory where reports will be saved.
        reference_file: Optional path to a JSONL file containing reference answers.
    """
    elo_system = EloRatingSystem()
    judge = PairwiseJudge()

    # Load reference answers if provided
    reference_answers = {}
    if reference_file:
        if reference_file.exists():
            logger.info(f"Loading reference answers from {reference_file}")
            reference_answers = load_model_outputs(reference_file)
        else:
            logger.error(f"Reference file not found: {reference_file}")
            return

    if report_dir:
        if report_dir.exists():
            shutil.rmtree(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Reports will be saved to {report_dir}")

    model_files = []

    if input_path.is_dir():
        model_files = list(input_path.glob("*.jsonl"))
        if len(model_files) < 2:
            logger.error("Directory must contain at least two .jsonl files.")
            return
        logger.info(f"Found {len(model_files)} model files in {input_path}")
    else:
        logger.error("Input must be a directory containing .jsonl files.")
        return

    pairs = list(itertools.combinations(model_files, 2))
    logger.info(f"Generated {len(pairs)} pairwise comparisons.")

    loaded_models = {}

    all_pairwise_results = []

    for file_a, file_b in pairs:
        model_a_name = file_a.stem
        model_b_name = file_b.stem

        if model_a_name not in loaded_models:
            loaded_models[model_a_name] = load_model_outputs(file_a)
        if model_b_name not in loaded_models:
            loaded_models[model_b_name] = load_model_outputs(file_b)

        outputs_a = loaded_models[model_a_name]
        outputs_b = loaded_models[model_b_name]

        pair_results = run_pairwise_comparison(
            model_a_name, outputs_a, model_b_name, outputs_b, elo_system, judge, reference_answers
        )
        all_pairwise_results.extend(pair_results)

    print("\n=== Final ELO Rankings ===")
    sorted_ratings = sorted(elo_system.ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, rating) in enumerate(sorted_ratings, 1):
        print(f"{rank}. {model}: {rating:.2f}")

    if report_dir:
        pairwise_file = report_dir / "pairwise_results.jsonl"
        with pairwise_file.open("w", encoding="utf-8") as f:
            for res in all_pairwise_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        logger.info(f"Pairwise results saved to {pairwise_file}")

        history_file = report_dir / "elo_history.jsonl"
        with history_file.open("w", encoding="utf-8") as f:
            for record in elo_system.history:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"ELO history saved to {history_file}")

        leaderboard_file = report_dir / "leaderboard.json"
        leaderboard_data = [
            {"rank": rank, "model": model, "rating": rating} for rank, (model, rating) in enumerate(sorted_ratings, 1)
        ]
        with leaderboard_file.open("w", encoding="utf-8") as f:
            json.dump(leaderboard_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Final leaderboard saved to {leaderboard_file}")


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment variables. Judge calls may fail.")

    parser = argparse.ArgumentParser(description="LLM-as-a-Judge ELO Ranking System")

    parser.add_argument("input_dir", type=Path, help="Path to directory containing .jsonl files for each model")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("report"),
        help="Directory to save report files (pairwise results, elo history, leaderboard)",
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Path to a JSONL file containing reference answers (optional)",
    )

    args = parser.parse_args()

    process_evaluations(args.input_dir, args.report_dir, args.reference)


if __name__ == "__main__":
    main()
