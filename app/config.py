import yaml

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Define some fallback default headings in case the local config is missing
HEADING_PROMPTS = {
    "title": [
        "In 5 words or less, list multiple concise and engaging titles for the following text",
        "In as few words as possible, list several short, attention grabbing titles for the following text",
        "In as few words as possible, list various potential headlines related to the following text",
        "Rephrase the following terms into a list of short pithy titles"
    ],
    "subtitle": [
        "state several terse and succinct statements describing the following text",
        "list several short, attention grabbing captions for the following text",
        "In 8 words or less, list various subtitles for the following text",
        "Rephrase the following statements into a list of short pithy subtitles"
    ],
    "description": [
        "Generate many short concise single-sentence descriptions of the following text",
        "List several brief, thematic explanations of the following text",
        "List multiple varied summaries of the following text",
        "Rephrase the following statements into a list of concise summaries"
    ],
}

# Define several different tag generation prompts for variety
TAGS_PROMPTS = [
    "With as few words as possible, list several related trending topics from the following text",
    "With as few words as possible, list high level ideas and themes of the following text",
    "With as few words as possible, list several tangentially related concepts to the following text",
]


# Define database settings class
class DatabaseSettings(BaseSettings):
    """Define database connection settings"""
    file: str = "sql_app.db"
    url: str = "sqlite:///./sql_app.db"
    connect_args: dict = {"check_same_thread": False}


# Define nested settings classes
class HeadingsSettings(BaseSettings):
    """Define default heading prompts"""
    title: list[str] = Field(default=HEADING_PROMPTS["title"], min_length=4)
    subtitle: list[str] = Field(default=HEADING_PROMPTS["subtitle"], min_length=4)
    description: list[str] = Field(default=HEADING_PROMPTS["description"], min_length=4)


class ModelSettings(BaseSettings):
    """Define available and default model types"""
    default: str = "google/gemma-3-1b-it"
    gemma: str = "google/gemma-3-1b-it"
    phi4: str = "microsoft/Phi-4-mini-instruct" 


class TransformersSettings(BaseSettings):
    """Define default keyword arguments for Transformers generation"""
    max_new_tokens: int = 128
    num_return_sequences: int = 1
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


# Define function default argument settings yaml class
class DefaultSettings(BaseSettings):
    """Define default keyword argument for core functions"""
    headings: HeadingsSettings = Field(default_factory=HeadingsSettings)
    tags: list[str] = Field(default=TAGS_PROMPTS, min_length=3)
    template: str = Field(default="{prompt}:\n\nText: {content}\n\n{delimiter}") 
    models: ModelSettings = Field(default_factory=ModelSettings)
    transformers: TransformersSettings = Field(default_factory=TransformersSettings)
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Define the application-level settings class which loads values from a .env file
class ApplicationSettings(BaseSettings):
    """Define all application-level settings"""
    # Load variables from a .env file if it exists
    model_config = SettingsConfigDict(env_file='.env')
    name: str = "NLP Service"
    version: str = "0.1.0"
    debug: bool = False

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    if Path("app/defaults.yaml").exists():
        # If the defaults.yaml file exists, load default settings from it
        defaults: DefaultSettings = DefaultSettings.from_yaml("app/defaults.yaml")
    else:
        defaults: DefaultSettings = Field(default_factory=DefaultSettings)


@lru_cache()
def get_settings() -> ApplicationSettings:
    return ApplicationSettings()
