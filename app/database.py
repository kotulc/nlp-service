from sqlmodel import create_engine, Session, SQLModel
from app.config import DatabaseSettings


# Define database engine
engine = create_engine(DATABASE_URL, echo=True, connect_args={"check_same_thread": False})


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session