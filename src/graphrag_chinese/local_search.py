import asyncio

from graphrag.query.structured_search.local_search.search import LocalSearch # pyright: ignore[reportMissingImports]

from graphrag_chinese.shared_resources import GraphRAGSharedResources


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


async def main() -> None:
    components = GraphRAGSharedResources()
    search_service = LocalSearchService(components)
    query = "What are the key challenges faced by Taipei City in managing its aging population?"
    result = await search_service.search(query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
