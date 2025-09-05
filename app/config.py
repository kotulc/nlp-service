import yaml

from functools import lru_cache
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
        "List several brief, terse explanations of the following text",
        "List multiple varied summaries of the following text",
        "Rephrase the following statements into a list of concise summaries"
    ],
}
TAGS_PROMPTS = [
    "With as few words as possible, list several related trending topics from the following text",
    "With as few words as possible, list high level ideas and themes of the following text",
    "With as few words as possible, list several tangentially related concepts to the following text",
]


# Define database settings class
class DatabaseSettings(BaseSettings):
    file: str = "sql_app.db"
    url: str = "sqlite:///./sql_app.db"
    connect_args: dict = {"check_same_thread": False}


# Define nested constants classes
class HeadingsSettings(BaseSettings):
    title: list[str] = Field(default=HEADING_PROMPTS["title"], min_length=4)
    subtitle: list[str] = Field(default=HEADING_PROMPTS["subtitle"], min_length=4)
    description: list[str] = Field(default=HEADING_PROMPTS["description"], min_length=4)


class ModelSettings(BaseSettings):
    default: str = "google/gemma-3-1b-it"
    gemma: str = "google/gemma-3-1b-it"
    phi4: str = "microsoft/Phi-4-mini-instruct" 

    
class GenerationSettings(BaseSettings):
    prompt: str = "Generate a concise summary"
    tone: str = "neutral"
    temperature: float = 0.7
    max_new_tokens: int = 32
    min_new_tokens: int = 16
    num_return_sequences: int = 1
    

class SummarySettings(BaseSettings):
    title = GenerationSettings
    subtitle = GenerationSettings
    description = GenerationSettings


class TransformersSettings(BaseSettings):
    max_new_tokens: int = 128
    num_return_sequences: int = 1
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


# Define function default argument settings yaml class
class DefaultSettings(BaseSettings):
    headings: HeadingsSettings
    models: ModelSettings
    summary: SummarySettings
    tags: list[str] = Field(default=TAGS_PROMPTS, min_length=3)
    template: str = Field(default="{prompt}:\n\nText: {content}\n\n{delimiter}") 
    transformers: TransformersSettings
    
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
