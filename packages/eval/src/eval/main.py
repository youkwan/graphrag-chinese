import argparse
import asyncio
import itertools
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from eval.elo import EloRatingSystem
from eval.judge import JudgeDecision, PairwiseJudge
from pydantic import BaseModel
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QA(BaseModel):
    """A single answer record loaded from JSONL."""
    question_id: str
    question: str
    response: str


class EloEvalRunner:
    """Runs pairwise judging and computes ELO rankings."""

    def __init__(
        self,
        *,
        judge: PairwiseJudge | None = None,
        elo_system: EloRatingSystem | None = None,
        max_concurrency: int = 8,
    ) -> None:
        """Initializes the runner.

        Args:
            judge: Pairwise LLM judge. If None, a default judge is created.
            elo_system: ELO rating system. If None, a default system is created.
            max_concurrency: Maximum number of concurrent judge calls.
        """
        self.judge = judge or PairwiseJudge()
        self.elo_system = elo_system or EloRatingSystem()
        self.max_concurrency = max(1, int(max_concurrency))

    def load_model_outputs(self, file_path: Path) -> list[QA]:
        """Loads model outputs from a JSONL file.

        Args:
            file_path: Path to the JSONL file.

        Returns:
            A list of QA objects.
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        qas: list[QA] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    question_id = str(data.get("question_id", ""))
                    question = str(data.get("question", ""))
                    response = str(data.get("response", ""))
                    if not question_id or not question or not response:
                        logger.warning(
                            f"{file_path} line {line_num} is missing 'question_id', 'question', or 'response', skipped."
                        )
                        continue
                    qas.append(QA(question_id=question_id, question=question, response=response))
                except Exception as e:
                    logger.error(f"{file_path} line {line_num} is not valid JSON: {e}")
                    raise ValueError(f"Invalid JSON on line {line_num} in {file_path}: {e}")

        return qas

    async def _judge_one(
        self,
        *,
        sem: asyncio.Semaphore,
        question: str,
        response_a: str,
        response_b: str,
    ) -> JudgeDecision:
        async with sem:
            return await self.judge.judge_async(question, response_a, response_b)

    def _score_from_winner(self, winner: str) -> float:
        if winner == "A":
            return 1.0
        if winner == "B":
            return 0.0
        return 0.5

    async def run_pairwise_comparison_async(
        self,
        *,
        name_a: str,
        qa_a: list[QA],
        name_b: str,
        qa_b: list[QA],
    ) -> list[dict[str, Any]]:
        """Runs pairwise comparison between two models using async judge calls.

        Judge calls are executed concurrently, while ELO updates are applied in a deterministic order.

        Args:
            name_a: Model A name.
            qa_a: Model A outputs.
            name_b: Model B name.
            qa_b: Model B outputs.

        Returns:
            A list of per-question comparison results.
        """
        if len(qa_a) != len(qa_b):
            logger.warning(
                f"Model {name_a} and {name_b} have different number of questions ({len(qa_a)} vs {len(qa_b)}), taking the smaller size."
            )
        n = min(len(qa_a), len(qa_b))
        sem = asyncio.Semaphore(self.max_concurrency)

        async def run_round(i: int, swapped: bool) -> tuple[int, bool, JudgeDecision | None, str | None]:
            question = qa_a[i].question
            response_a = qa_a[i].response
            response_b = qa_b[i].response
            try:
                if not swapped:
                    decision = await self._judge_one(
                        sem=sem, question=question, response_a=response_a, response_b=response_b
                    )
                else:
                    decision = await self._judge_one(
                        sem=sem, question=question, response_a=response_b, response_b=response_a
                    )
                return i, swapped, decision, None
            except Exception as e:
                return i, swapped, None, str(e)

        tasks = [asyncio.create_task(run_round(i, swapped)) for i in range(n) for swapped in (False, True)]
        round_results = await asyncio.gather(*tasks)
        round_results.sort(key=lambda x: (x[0], x[1]))  # deterministic: round1 then swapped

        results: list[dict[str, Any]] = []

        for i, swapped, decision, err in round_results:
            question_id = qa_a[i].question_id if not swapped else qa_b[i].question_id
            question = qa_a[i].question
            response_a = qa_a[i].response
            response_b = qa_b[i].response

            if err or decision is None:
                if not swapped:
                    logger.error(f"Scoring failed: {name_a} vs {name_b} (question {question_id}): {err}")
                else:
                    logger.error(f"Scoring failed: {name_b} vs {name_a} (swapped, question {question_id}): {err}")
                continue

            if not swapped:
                score_a = self._score_from_winner(decision.winner)
                self.elo_system.update_ratings(name_a, name_b, score_a)
                results.append(
                    {
                        "question_id": question_id,
                        "model_a": name_a,
                        "model_b": name_b,
                        "response_a": response_a,
                        "response_b": response_b,
                        "judge_decision": decision.winner,
                        "judge_explanation": decision.explanation,
                        "swapped": False,
                    }
                )
            else:
                # The judge saw (response_b as A, response_a as B).
                # Convert the judge winner into "who actually won in original naming".
                if decision.winner == "A":
                    score_a = 0.0
                    real_winner = "B"
                elif decision.winner == "B":
                    score_a = 1.0
                    real_winner = "A"
                else:
                    score_a = 0.5
                    real_winner = "Tie"

                self.elo_system.update_ratings(name_a, name_b, score_a)
                results.append(
                    {
                        "question_id": question_id,
                        "model_a": name_b,
                        "model_b": name_a,
                        "response_a": response_b,
                        "response_b": response_a,
                        "judge_decision": decision.winner,
                        "judge_explanation": decision.explanation,
                        "swapped": True,
                        "real_winner": real_winner,
                    }
                )

        return results

    async def process_evaluations_async(self, input_path: Path, report_dir: Path | None = None) -> None:
        """Reads comparisons from a directory, updates ELO, and saves detailed reports.

        Args:
            input_path: Path to a directory containing JSONL files.
            report_dir: Path to the directory where reports will be saved.
        """
        if report_dir:
            if report_dir.exists():
                shutil.rmtree(report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Reports will be saved to: {report_dir}")

        if not input_path.is_dir():
            logger.error("Input must be a folder containing .jsonl files.")
            return

        model_files = list(input_path.glob("*.jsonl"))
        if len(model_files) < 2:
            logger.error("The folder must contain at least two .jsonl files.")
            return
        logger.info(f"Found {len(model_files)} model output files in {input_path}")

        pairs = list(itertools.combinations(model_files, 2))
        logger.info(f"Generated {len(pairs)} pairwise comparisons.")

        loaded_models: dict[str, list[QA]] = {}
        all_pairwise_results: list[dict[str, Any]] = []

        for file_a, file_b in pairs:
            model_a_name = file_a.stem
            model_b_name = file_b.stem

            if model_a_name not in loaded_models:
                loaded_models[model_a_name] = self.load_model_outputs(file_a)
            if model_b_name not in loaded_models:
                loaded_models[model_b_name] = self.load_model_outputs(file_b)

            outputs_a = loaded_models[model_a_name]
            outputs_b = loaded_models[model_b_name]

            pair_results = await self.run_pairwise_comparison_async(
                name_a=model_a_name, qa_a=outputs_a, name_b=model_b_name, qa_b=outputs_b
            )
            all_pairwise_results.extend(pair_results)

        print("\n=== Final ELO Leaderboard ===")
        sorted_ratings = sorted(self.elo_system.ratings.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, rating) in enumerate(sorted_ratings, 1):
            print(f"{rank}. {model}: {rating:.2f}")

        if report_dir:
            pairwise_file = report_dir / "pairwise_results.jsonl"
            with pairwise_file.open("w", encoding="utf-8") as f:
                for res in all_pairwise_results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            logger.info(f"Pairwise comparison results saved to: {pairwise_file}")

            history_file = report_dir / "elo_history.jsonl"
            with history_file.open("w", encoding="utf-8") as f:
                for record in self.elo_system.history:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"ELO history saved to: {history_file}")

            leaderboard_file = report_dir / "leaderboard.json"
            leaderboard_data = [
                {"rank": rank, "model": model, "rating": rating}
                for rank, (model, rating) in enumerate(sorted_ratings, 1)
            ]
            with leaderboard_file.open("w", encoding="utf-8") as f:
                json.dump(leaderboard_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Leaderboard saved to: {leaderboard_file}")


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set (judge calls may fail).")

    parser = argparse.ArgumentParser(description="LLM-as-a-Judge ELO Ranking System (Async Version)")

    parser.add_argument("--input_dir", type=Path, help="Path to directory containing .jsonl model outputs")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("report"),
        help="Directory to save report files (pairwise results / elo history / leaderboard)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent judge calls (avoid triggering rate limits)",
    )

    args = parser.parse_args()

    runner = EloEvalRunner(max_concurrency=args.max_concurrency)
    asyncio.run(runner.process_evaluations_async(args.input_dir, args.report_dir))


if __name__ == "__main__":
    main()
