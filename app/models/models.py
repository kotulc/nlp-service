import uuid

from sqlmodel import Field, SQLModel


# Define base content class
class BaseSQLModel(SQLModel):
    id: int | None = Field(default=None, primary_key=True)


class Document(BaseSQLModel, table=True):
    """Markdown document"""
    name: str = Field(min_length=1, max_length=255, index=True)
    title: str = Field(min_length=1, max_length=255)
    published: str | None = Field(default=None, min_length=1, max_length=255)
    source: str | None = Field(default=None, min_length=1, max_length=255)


class Section(BaseSQLModel, table=True):
    """Document section headings"""
    name: str = Field(min_length=1, max_length=255, index=True)
    title: str = Field(min_length=1, max_length=255)
    document: int | None = Field(default=None, foreign_key="document.id")
    parent: int | None = Field(default=None, foreign_key="section.id")


class Content(BaseSQLModel):
    """Document, section or paragraph"""
    content: str = Field(min_length=1, max_length=255)
    document: int | None = Field(default=None, foreign_key="document.id")
    section: int | None = Field(default=None, foreign_key="section.id")
    parent: int | None = Field(default=None, foreign_key="content.id")


class Metric(BaseSQLModel):
    """Computed or derived composite metric"""
    category: str = Field(min_length=1, max_length=32, index=True)
    content: int = Field(foreign_key='content.id')
    score: float


class Summary(BaseSQLModel):
    """Generated content title, subtitle, or outline scored by relevance"""
    summary: str = Field(min_length=1, max_length=255)
    category: str = Field(min_length=1, max_length=32, index=True)
    content: int = Field(foreign_key='content.id')
    selected: bool = Field(default=False)
    score: float


class Tag(BaseSQLModel):
    """A single unique content relevance scored tag"""
    tag: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=32, index=True) 
    content: int = Field(foreign_key='content.id')
    selected: bool = Field(default=False)
    score: float
