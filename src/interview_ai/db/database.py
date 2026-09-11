from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from interview_ai.config import config

engine = create_async_engine(config.database_url, echo=config.is_debug)

session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)


async def get_db():
    async with session_factory() as session:
        yield session
