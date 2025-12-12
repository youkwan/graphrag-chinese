from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
# from dotenv import load_dotenv

INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "output"
COMMUNITY_REPORT_TABLE = "community_reports"
ENTITY_TABLE = "entities"
COMMUNITY_TABLE = "communities"
RELATIONSHIP_TABLE = "relationships"
TEXT_UNIT_TABLE = "text_units"
COMMUNITY_LEVEL = 2
ENV_FILE_PATH = INPUT_DIR.parents[0] / ".env"


class Settings(BaseSettings):
    graphrag_api_key: str
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH)


# Singleton instance
Settings = Settings()


def main() -> None:
    print(Settings.graphrag_api_key)


if __name__ == "__main__":
    main()
