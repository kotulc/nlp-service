import yaml

from pydantic_settings import BaseSettings


# Define application-level settings class inheriting from BaseSettings
class ApplicationSettings(BaseSettings):
    name: str
    debug: bool = False
    defaults_path: str = "defaults.yaml"


# Define application-level settings class inheriting from BaseSettings
class DatabaseSettings(BaseSettings):
    db_url: str = "sqlite:///./sql_app.db"


# Define application-level settings class inheriting from BaseSettings
class DefaultSettings(BaseSettings):
    generation: dict[str, dict]
    models: dict[str, str]
    summary: dict[str, dict]
    templates: dict[str, str]
    
    @classmethod
    def from_yaml(cls, path: str = "config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
