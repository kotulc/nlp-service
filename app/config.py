import yaml

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


# Define database settings class
class DatabaseSettings(BaseSettings):
    file: str = "sql_app.db"
    url: str = "sqlite:///./sql_app.db"
    connect_args: dict = {"check_same_thread": False}


# Define function constants class
class DefaultSettings(BaseSettings):
    generation: dict[str, dict]
    models: dict[str, str]
    summary: dict[str, dict]
    templates: dict[str, str]
    
    @classmethod
    def from_yaml(cls, path: str="defaults.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Define application-level settings class inheriting from BaseSettings
class ApplicationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    name: str = "NLP Service"
    version: str = "0.1.0"
    debug: bool = False

    database = DatabaseSettings
    defaults = DefaultSettings.from_yaml()


@lru_cache()
def get_settings() -> ApplicationSettings:
    return ApplicationSettings()
