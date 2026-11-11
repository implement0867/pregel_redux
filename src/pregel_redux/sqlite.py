from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID

from pydantic import TypeAdapter
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from pregel_redux.checkpoints import Checkpoint, CheckpointTuple, PendingWrite


class Base(DeclarativeBase):
    pass


class CheckpointRow(Base):
    __tablename__ = "checkpoints"

    thread_id: Mapped[str] = mapped_column(primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(primary_key=True)
    checkpoint_id: Mapped[UUID] = mapped_column(primary_key=True)
    checkpoint: Mapped[str] = mapped_column()


class WriteRow(Base):
    __tablename__ = "writes"

    thread_id: Mapped[str] = mapped_column(primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(primary_key=True)
    checkpoint_id: Mapped[UUID] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(primary_key=True)
    task_id: Mapped[str] = mapped_column(primary_key=True)
    write: Mapped[str] = mapped_column()


@dataclass(frozen=True)
class SqliteCheckpointer:
    engine: AsyncEngine

    async def setup(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def aput(self, thread_id: str, checkpoint: Checkpoint) -> None:
        async with AsyncSession(self.engine) as session, session.begin():
            session.add(
                CheckpointRow(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint.checkpoint_ns,
                    checkpoint_id=checkpoint.id,
                    checkpoint=checkpoint.model_dump_json(),
                )
            )

    async def aput_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        async with AsyncSession(self.engine) as session, session.begin():
            await session.merge(
                WriteRow(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    kind=write.kind.value,
                    task_id=write.task_id or "",
                    write=write.model_dump_json(),
                )
            )

    async def read_tuple(self, session: AsyncSession, row: CheckpointRow) -> CheckpointTuple:
        writes = await session.scalars(
            select(WriteRow.write)
            .where(
                WriteRow.thread_id == row.thread_id,
                WriteRow.checkpoint_ns == row.checkpoint_ns,
                WriteRow.checkpoint_id == row.checkpoint_id,
            )
            .order_by(WriteRow.kind, WriteRow.task_id)
        )
        adapter = TypeAdapter[PendingWrite](PendingWrite)
        return CheckpointTuple(
            checkpoint=Checkpoint.model_validate_json(row.checkpoint),
            pending_writes=tuple(adapter.validate_json(write) for write in writes),
        )

    async def aget_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None
    ) -> CheckpointTuple | None:
        query = select(CheckpointRow).where(
            CheckpointRow.thread_id == thread_id, CheckpointRow.checkpoint_ns == checkpoint_ns
        )
        if checkpoint_id is not None:
            query = query.where(CheckpointRow.checkpoint_id == checkpoint_id)
        async with AsyncSession(self.engine) as session:
            row = await session.scalar(query.order_by(CheckpointRow.checkpoint_id.desc()).limit(1))
            return None if row is None else await self.read_tuple(session, row)

    async def alist(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> AsyncIterator[CheckpointTuple]:
        query = select(CheckpointRow).where(
            CheckpointRow.thread_id == thread_id, CheckpointRow.checkpoint_ns == checkpoint_ns
        )
        if before is not None:
            query = query.where(CheckpointRow.checkpoint_id < before)
        async with AsyncSession(self.engine) as session:
            rows = await session.scalars(query.order_by(CheckpointRow.checkpoint_id.desc()).limit(limit))
            for row in rows.all():
                yield await self.read_tuple(session, row)

    async def adelete_thread(self, thread_id: str) -> None:
        async with AsyncSession(self.engine) as session, session.begin():
            await session.execute(delete(WriteRow).where(WriteRow.thread_id == thread_id))
            await session.execute(delete(CheckpointRow).where(CheckpointRow.thread_id == thread_id))
