import asyncio
import json
import os
from pathlib import Path
from typing import Any

from graphrag.query.structured_search.local_search.search import LocalSearch # pyright: ignore[reportMissingImports]

from graphrag_chinese.shared_resources import GraphRAGSharedResources
from graphrag_chinese.constants import QUESTIONS_FILE, ANSWERS_FOLDER

class LocalSearchService:
    """Wrapper for local search functionality."""

    def __init__(self, components: GraphRAGSharedResources) -> None:
        """Create local search service using shared components.

        Args:
            components (GraphRAGSharedResources): Initialized shared components.
        """

        self.search_engine = LocalSearch(
            model=components.chat_model,
            context_builder=components.context_builder,
            tokenizer=components.tokenizer,
            model_params=components.model_params,
            context_builder_params=components.local_context_params,
            response_type="multiple paragraphs",
        )

    async def search(self, query: str) -> str:
        """Execute query and return text result.

        Args:
            query (str): Query content.

        Returns:
            str: Text content of the query response.
        """

        result = await self.search_engine.search(query)
        return result.response


def _load_questions_jsonl(questions_path: Path) -> list[dict[str, Any]]:
    """Load benchmark questions from a JSONL file.

    The expected schema per line is:
        {"question_id": "<str|int>", "question": "<str>"}

    Args:
        questions_path: Path to the questions JSONL file.

    Returns:
        A list of question records.

    Raises:
        FileNotFoundError: If the questions file does not exist.
        ValueError: If a line is invalid or required fields are missing.
    """

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions: list[dict[str, Any]] = []
    with questions_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} in {questions_path}") from e

            question_id = record.get("question_id")
            question = record.get("question")
            if question_id is None or not str(question_id).strip():
                raise ValueError(f"Line {line_num} missing 'question_id' in {questions_path}")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_num} missing 'question' in {questions_path}")

            questions.append({"question_id": str(question_id), "question": question})

    return questions


async def generate_answers(name: str) -> Path:
    """Generate responses for each benchmark question and save to `benchmarks/answers`.
    and write a JSONL file where each line includes:
        - question_id: question_id (string, used as the stable evaluation key)
        - question: the original question text
        - response: model response text

    Returns:
        Path: The output JSONL file path.
    """

    questions_path = Path(QUESTIONS_FILE)
    output_dir = ANSWERS_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.jsonl"

    questions = _load_questions_jsonl(questions_path)

    components = GraphRAGSharedResources()
    search_service = LocalSearchService(components)

    max_concurrency = int(os.getenv("GRAPHRAG_BENCHMARK_CONCURRENCY", "8"))
    max_concurrency = max(1, max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(index: int, q: dict[str, Any]) -> tuple[int, dict[str, str]]:
        """Run local search for one question (bounded by semaphore)."""
        async with semaphore:
            question = q["question"]
            response_text = await search_service.search(question)
            out_record: dict[str, str] = {
                "question_id": q["question_id"],
                "question": question,
                "response": response_text,
            }
            return index, out_record

    tasks = [
        asyncio.create_task(_run_one(idx, q))
        for idx, q in enumerate(questions, 1)
    ]

    results: dict[int, dict[str, str]] = {}
    completed = 0
    try:
        for task in asyncio.as_completed(tasks):
            idx, out_record = await task
            results[idx] = out_record
            completed += 1
            print(f"Completed {completed}/{len(questions)}: question_id={out_record['question_id']}")
    except Exception:
        for t in tasks:
            if not t.done():
                t.cancel()
        raise

    with output_path.open("w", encoding="utf-8") as f:
        for idx in range(1, len(questions) + 1):
            f.write(json.dumps(results[idx], ensure_ascii=False) + "\n")

    print(f"Output written to: {output_path}")
    return output_path


async def main() -> None:
    await generate_answers("default")


if __name__ == "__main__":
    asyncio.run(main())
