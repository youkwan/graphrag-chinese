from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

def find_project_root(start_path: Path, marker: str = ".git") -> Path:
    """Find the project root directory by searching for a marker file."""
    for p in start_path.parents:
        if (p / marker).exists():
            return p
    raise ValueError(f"Project root not found in {start_path}")

PROJECT_ROOT = find_project_root(Path(__file__))

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
COMMUNITY_REPORT_TABLE = "community_reports"
ENTITY_TABLE = "entities"
COMMUNITY_TABLE = "communities"
RELATIONSHIP_TABLE = "relationships"
TEXT_UNIT_TABLE = "text_units"
COMMUNITY_LEVEL = 2

ENV_FILE = PROJECT_ROOT / "data" / ".env"

QUESTIONS_FILE = PROJECT_ROOT / "benchmarks" / "questions" / "questions_a.jsonl"
ANSWERS_FOLDER = PROJECT_ROOT / "benchmarks" / "answers"


class AppSettings(BaseSettings):
    graphrag_api_key: str = ""
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")


APP_SETTINGS = AppSettings()


def main() -> None:
    print(APP_SETTINGS.graphrag_api_key)


if __name__ == "__main__":
    main()
