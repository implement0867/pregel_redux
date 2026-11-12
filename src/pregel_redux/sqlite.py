import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from uuid import UUID

from pydantic import TypeAdapter
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

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


def read_tuple(session: Session, row: CheckpointRow) -> CheckpointTuple:
    writes = session.scalars(
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


@dataclass(frozen=True)
class SqliteCheckpointer:
    engine: Engine

    def setup(self) -> None:
        Base.metadata.create_all(self.engine)

    def put(self, thread_id: str, checkpoint: Checkpoint) -> None:
        with Session(self.engine) as session, session.begin():
            session.add(
                CheckpointRow(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint.checkpoint_ns,
                    checkpoint_id=checkpoint.id,
                    checkpoint=checkpoint.model_dump_json(),
                )
            )

    def put_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        with Session(self.engine) as session, session.begin():
            session.merge(
                WriteRow(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    kind=write.kind.value,
                    task_id=write.task_id or "",
                    write=write.model_dump_json(),
                )
            )

    def get_tuple(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None) -> CheckpointTuple | None:
        query = select(CheckpointRow).where(
            CheckpointRow.thread_id == thread_id, CheckpointRow.checkpoint_ns == checkpoint_ns
        )
        if checkpoint_id is not None:
            query = query.where(CheckpointRow.checkpoint_id == checkpoint_id)
        with Session(self.engine) as session:
            row = session.scalar(query.order_by(CheckpointRow.checkpoint_id.desc()).limit(1))
            return None if row is None else read_tuple(session, row)

    def list(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> Iterator[CheckpointTuple]:
        query = select(CheckpointRow).where(
            CheckpointRow.thread_id == thread_id, CheckpointRow.checkpoint_ns == checkpoint_ns
        )
        if before is not None:
            query = query.where(CheckpointRow.checkpoint_id < before)
        with Session(self.engine) as session:
            rows = session.scalars(query.order_by(CheckpointRow.checkpoint_id.desc()).limit(limit)).all()
            yield from [read_tuple(session, row) for row in rows]

    def delete_thread(self, thread_id: str) -> None:
        with Session(self.engine) as session, session.begin():
            session.execute(delete(WriteRow).where(WriteRow.thread_id == thread_id))
            session.execute(delete(CheckpointRow).where(CheckpointRow.thread_id == thread_id))

    async def aput(self, thread_id: str, checkpoint: Checkpoint) -> None:
        await asyncio.to_thread(self.put, thread_id, checkpoint)

    async def aput_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        await asyncio.to_thread(self.put_writes, thread_id, checkpoint_ns, checkpoint_id, write)

    async def aget_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None
    ) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, thread_id, checkpoint_ns, checkpoint_id)

    async def alist(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> AsyncIterator[CheckpointTuple]:
        found = await asyncio.to_thread(lambda: tuple(self.list(thread_id, checkpoint_ns, before=before, limit=limit)))
        for saved in found:
            yield saved

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)
