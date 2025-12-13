from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from graphrag.config.enums import ModelType # pyright: ignore[reportMissingImports]
from graphrag.config.models.language_model_config import LanguageModelConfig # pyright: ignore[reportMissingImports]
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig # pyright: ignore[reportMissingImports]   
from graphrag.language_model.manager import ModelManager # pyright: ignore[reportMissingImports]
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey # pyright: ignore[reportMissingImports]
from graphrag.query.indexer_adapters import ( # pyright: ignore[reportMissingImports]
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.local_search.mixed_context import ( # pyright: ignore[reportMissingImports]
    LocalSearchMixedContext,
)
from graphrag.tokenizer.get_tokenizer import get_tokenizer # pyright: ignore[reportMissingImports]
from graphrag.vector_stores.lancedb import LanceDBVectorStore # pyright: ignore[reportMissingImports]

from graphrag_chinese.constants import (
    COMMUNITY_LEVEL,
    COMMUNITY_REPORT_TABLE,
    COMMUNITY_TABLE,
    ENTITY_TABLE,
    OUTPUT_DIR,
    RELATIONSHIP_TABLE,
    TEXT_UNIT_TABLE,
    APP_SETTINGS,
)


@dataclass
class GraphRAGSharedResources:
    """Initialization wrapper for GraphRAG shared resources.

    Args:
        input_dir (Path): Data input path.
        community_level (int): Community level setting.
        chat_model_config (LanguageModelConfig | None): Custom chat model config; defaults to default if not provided.
        embedding_model_config (LanguageModelConfig | None): Custom embedding model config; defaults to default if not provided.
        local_context_params (dict[str, Any] | None): Custom context builder parameters; defaults to default if not provided.
        model_params (dict[str, Any] | None): Custom model generation parameters; defaults to default if not provided.
        api_key (str | None): Explicitly specified API key.
    """

    input_dir: Path = OUTPUT_DIR
    community_level: int = COMMUNITY_LEVEL
    chat_model_config: LanguageModelConfig | None = None
    embedding_model_config: LanguageModelConfig | None = None
    local_context_params: dict[str, Any] | None = None
    model_params: dict[str, Any] | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        self._lancedb_uri = self.input_dir / "lancedb"
        self.entities = None
        self.relationships = None
        self.reports = None
        self.text_units = None
        self.description_embedding_store = None
        self.chat_model = None
        self.text_embedder = None
        self.tokenizer = None
        self.context_builder = None
        self.local_context_params = self.local_context_params or {}
        self.model_params = self.model_params or {}
        self._load_data()
        self._setup_models()
        self._setup_context()
        self._setup_params()

    def _load_data(self) -> None:
        entity_df = pd.read_parquet(self.input_dir / f"{ENTITY_TABLE}.parquet")
        community_df = pd.read_parquet(self.input_dir / f"{COMMUNITY_TABLE}.parquet")
        self.entities = read_indexer_entities(entity_df, community_df, self.community_level)

        self.description_embedding_store = LanceDBVectorStore(
            vector_store_schema_config=VectorStoreSchemaConfig(index_name="default-entity-description")
        )
        self.description_embedding_store.connect(db_uri=str(self._lancedb_uri))

        relationship_df = pd.read_parquet(self.input_dir / f"{RELATIONSHIP_TABLE}.parquet")
        self.relationships = read_indexer_relationships(relationship_df)

        report_df = pd.read_parquet(self.input_dir / f"{COMMUNITY_REPORT_TABLE}.parquet")
        self.reports = read_indexer_reports(report_df, community_df, self.community_level)

        text_unit_df = pd.read_parquet(self.input_dir / f"{TEXT_UNIT_TABLE}.parquet")
        self.text_units = read_indexer_text_units(text_unit_df)

    def _setup_models(self) -> None:
        api_key = self.api_key or APP_SETTINGS.graphrag_api_key
        chat_config = self.chat_model_config or LanguageModelConfig(
            api_key=api_key,
            type=ModelType.Chat,
            model_provider="azure",
            model="gpt-4o-mini",
            max_retries=20,
            api_base="https://aihai-ai01.openai.azure.com",
            api_version="2024-12-01-preview",
        )
        self.chat_model = ModelManager().get_or_create_chat_model(
            name="local_search",
            model_type=ModelType.Chat,
            config=chat_config,
        )

        embedding_config = self.embedding_model_config or LanguageModelConfig(
            api_key=api_key,
            type=ModelType.Embedding,
            model_provider="azure",
            model="text-embedding-3-small",
            max_retries=20,
            api_base="https://aihai-ai01.openai.azure.com",
            api_version="2024-12-01-preview",
        )
        self.text_embedder = ModelManager().get_or_create_embedding_model(
            name="local_search_embedding",
            model_type=ModelType.Embedding,
            config=embedding_config,
        )
        self.tokenizer = get_tokenizer(chat_config)

    def _setup_context(self) -> None:
        self.context_builder = LocalSearchMixedContext(
            community_reports=self.reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            covariates=None,
            entity_text_embeddings=self.description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=self.text_embedder,
            tokenizer=self.tokenizer,
        )

    def _setup_params(self) -> None:
        default_local_context_params: dict[str, Any] = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            "max_tokens": 12_000,
        }
        default_model_params: dict[str, Any] = {
            "max_tokens": 8_000,
            "temperature": 0.0,
        }
        merged_local_context = {**default_local_context_params, **(self.local_context_params or {})}
        merged_model_params = {**default_model_params, **(self.model_params or {})}
        self.local_context_params = merged_local_context
        self.model_params = merged_model_params
