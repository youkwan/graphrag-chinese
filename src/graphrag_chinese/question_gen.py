import asyncio
from typing import Any

from graphrag.query.question_gen.local_gen import LocalQuestionGen

from graphrag_chinese.shared_resources import GraphRAGSharedResources


class LocalQuestionGenService:
    """Wrapper for local question generation."""

    def __init__(self, components: GraphRAGSharedResources) -> None:
        """Create question generation service using shared components.

        Args:
            components (GraphRAGSharedResources): Initialized shared components.
        """

        self.question_generator = LocalQuestionGen(
            model=components.chat_model,
            context_builder=components.context_builder,
            tokenizer=components.tokenizer,
            model_params=components.model_params,
            context_builder_params=components.local_context_params,
        )

    async def generate(
        self,
        question_history: list | None = None,
        context_data: Any | None = None,
        question_count: int = 5,
    ) -> str:
        """Generate a list of suggested questions.

        Args:
            question_history (Optional[list]): Previous Q&A history.
            context_data (Optional[Any]): Additional contextual data.
            question_count (int): Number of questions to generate.

        Returns:
            str: The generated question list as text.
        """

        question_history = question_history or []
        candidate_questions = await self.question_generator.agenerate(
            question_history=question_history,
            context_data=context_data,
            question_count=question_count,
        )
        return candidate_questions.response


async def main() -> None:
    components = GraphRAGSharedResources()
    question_service = LocalQuestionGenService(components)
    candidate_questions = await question_service.generate(question_history=[], context_data=None, question_count=30)
    print(candidate_questions)

if __name__ == "__main__":
    asyncio.run(main())
