"""Message archive service for preserving complete chat history.

This service archives messages BEFORE they can be compressed by SummarizationMiddleware,
allowing full history reconstruction for UI rendering even after the checkpoint
has been summarized.
"""

from typing import Any

import structlog
from sqlalchemy import func, select, Text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.orm import MessageArchive

logger = structlog.get_logger(__name__)


class MessageArchiveService:
    """Service for archiving and retrieving complete message history."""

    def _is_summary_message(self, msg_data: dict) -> bool:
        """Check if message is a summary from SummarizationMiddleware."""
        content = msg_data.get("content", {})
        if isinstance(content, str):
            return content.startswith("Here is a summary of the conversation")
        if isinstance(content, dict):
            text = content.get("text", "")
            return text.startswith("Here is a summary of the conversation")
        return False

    async def archive_messages(
        self,
        session: AsyncSession,
        thread_id: str,
        messages: list[Any],
    ) -> int:
        """
        Archive messages from LangGraph state.
        
        CRITICAL DESIGN:
        - Uses message_id as the deduplication key (not message_index!)
        - INSERT only, never UPDATE existing messages
        - This prevents SummarizationMiddleware compressed messages from
          overwriting the original history
        
        Args:
            session: Database session
            thread_id: Thread identifier
            messages: List of LangGraph messages to archive
            
        Returns:
            Number of NEW messages archived (excludes already-existing)
        """
        if not messages:
            return 0

        # First, get existing message_ids to avoid duplicates
        existing_ids = await self._get_existing_message_ids(session, thread_id)
        
        # Also get current max index for new messages
        max_index = await self._get_max_index(session, thread_id)

        archived_count = 0
        skipped_existing = 0
        skipped_summary = 0
        
        for msg in messages:
            msg_data = self._serialize_message(msg, 0)  # index will be set later
            
            # Skip if we can't serialize
            if not msg_data:
                continue
            
            # Skip summary messages - they replace real history
            if self._is_summary_message(msg_data):
                skipped_summary += 1
                continue
            
            msg_id = msg_data.get("id")
            
            # Skip if message already exists (by message_id)
            if msg_id and msg_id in existing_ids:
                skipped_existing += 1
                continue
            
            # Assign new index (append to end)
            max_index += 1
            
            # INSERT only - no UPSERT, no UPDATE
            stmt = insert(MessageArchive).values(
                thread_id=thread_id,
                message_index=max_index,
                message_id=msg_id,
                message_type=msg_data["type"],
                content=msg_data["content"],
                tool_calls=msg_data.get("tool_calls"),
                tool_call_id=msg_data.get("tool_call_id"),
                metadata_json=msg_data.get("metadata"),
            ).on_conflict_do_nothing()  # If somehow duplicated, just skip
            
            await session.execute(stmt)
            existing_ids.add(msg_id)  # Track for this batch
            archived_count += 1

        await session.commit()
        logger.info(
            "archived_messages",
            thread_id=thread_id,
            new_count=archived_count,
            skipped_existing=skipped_existing,
            skipped_summary=skipped_summary,
        )
        return archived_count

    async def _get_existing_message_ids(
        self,
        session: AsyncSession,
        thread_id: str,
    ) -> set[str]:
        """Get all existing message IDs for a thread."""
        from sqlalchemy import select
        stmt = (
            select(MessageArchive.message_id)
            .where(MessageArchive.thread_id == thread_id)
            .where(MessageArchive.message_id.isnot(None))
        )
        result = await session.scalars(stmt)
        return set(result.all())

    async def _get_max_index(
        self,
        session: AsyncSession,
        thread_id: str,
    ) -> int:
        """Get the maximum message_index for a thread, or -1 if empty."""
        stmt = (
            select(func.max(MessageArchive.message_index))
            .where(MessageArchive.thread_id == thread_id)
        )
        result = await session.scalar(stmt)
        return result if result is not None else -1

    async def get_full_history(
        self,
        session: AsyncSession,
        thread_id: str,
    ) -> list[dict]:
        """
        Get complete archived history for a thread.
        
        NOTE: Filters out any summary messages that may have leaked into archive
        from older buggy code.
        
        Args:
            session: Database session
            thread_id: Thread identifier
            
        Returns:
            List of messages in LangGraph-compatible format
        """
        stmt = (
            select(MessageArchive)
            .where(MessageArchive.thread_id == thread_id)
            .order_by(MessageArchive.message_index)
        )
        result = await session.scalars(stmt)

        messages = []
        for m in result:
            msg_dict = self._deserialize_message(m)
            # Filter out summary messages (defensive - shouldn't exist in new archives)
            if not self._is_summary_message_content(msg_dict.get("content", "")):
                messages.append(msg_dict)
        
        return messages

    def _is_summary_message_content(self, content: Any) -> bool:
        """Check if content indicates a summary message."""
        if isinstance(content, str):
            return content.startswith("Here is a summary of the conversation")
        return False

    async def cleanup_summary_messages(
        self,
        session: AsyncSession,
        thread_id: str,
    ) -> int:
        """
        Remove summary messages from archive (for fixing corrupted threads).
        
        Returns:
            Number of messages deleted
        """
        from sqlalchemy import delete, or_
        
        # Delete messages where content starts with summary prefix
        # This is a bit tricky with JSONB, need to handle both string and dict content
        stmt = (
            delete(MessageArchive)
            .where(MessageArchive.thread_id == thread_id)
            .where(
                or_(
                    MessageArchive.content["text"].astext.startswith("Here is a summary of the conversation"),
                    # For string content stored directly
                    func.cast(MessageArchive.content, Text).startswith('{"text": "Here is a summary')
                )
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        
        deleted = result.rowcount or 0
        if deleted > 0:
            logger.info(
                "cleaned_up_summary_messages",
                thread_id=thread_id,
                deleted_count=deleted,
            )
        return deleted

    async def get_archive_count(
        self,
        session: AsyncSession,
        thread_id: str,
    ) -> int:
        """Get count of archived messages for a thread."""
        stmt = (
            select(func.count())
            .select_from(MessageArchive)
            .where(MessageArchive.thread_id == thread_id)
        )
        return await session.scalar(stmt) or 0

    async def has_archive(
        self,
        session: AsyncSession,
        thread_id: str,
    ) -> bool:
        """Check if a thread has any archived messages."""
        count = await self.get_archive_count(session, thread_id)
        return count > 0

    async def delete_archive(
        self,
        session: AsyncSession,
        thread_id: str,
    ) -> int:
        """Delete all archived messages for a thread."""
        from sqlalchemy import delete
        
        stmt = delete(MessageArchive).where(MessageArchive.thread_id == thread_id)
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount or 0

    def _serialize_message(self, msg: Any, index: int) -> dict | None:
        """
        Serialize a LangGraph message for storage.
        
        Handles different message formats:
        - Pydantic models (BaseMessage subclasses)
        - Dicts (from checkpoint serialization)
        """
        try:
            # Handle Pydantic models
            if hasattr(msg, "model_dump"):
                data = msg.model_dump()
            elif hasattr(msg, "dict"):
                data = msg.dict()
            elif isinstance(msg, dict):
                data = msg
            else:
                logger.warning(
                    "unknown_message_type",
                    msg_type=type(msg).__name__,
                    index=index,
                )
                return None

            # Normalize content to dict for JSONB storage
            content = data.get("content")
            if isinstance(content, str):
                content = {"text": content}
            elif content is None:
                content = {}

            return {
                "id": data.get("id"),
                "type": data.get("type", "unknown"),
                "content": content,
                "tool_calls": data.get("tool_calls"),
                "tool_call_id": data.get("tool_call_id"),
                "metadata": {
                    "created_at": data.get("created_at"),
                    "additional_kwargs": data.get("additional_kwargs"),
                    "response_metadata": data.get("response_metadata"),
                    "name": data.get("name"),
                },
            }
        except Exception as e:
            logger.error(
                "serialize_message_failed",
                index=index,
                error=str(e),
            )
            return None

    def _deserialize_message(self, archive: MessageArchive) -> dict:
        """
        Deserialize archived message back to LangGraph-compatible format.
        
        Returns a dict that matches the format expected by the frontend's
        reconstructHistoryFromMessages function.
        """
        # Extract text content if stored as dict
        content = archive.content
        if isinstance(content, dict) and "text" in content:
            content = content["text"]

        metadata = archive.metadata_json or {}
        
        result = {
            "id": archive.message_id,
            "type": archive.message_type,
            "content": content,
            "created_at": (
                archive.created_at.isoformat() if archive.created_at else None
            ),
        }
        
        # Add optional fields
        if archive.tool_calls:
            result["tool_calls"] = archive.tool_calls
        if archive.tool_call_id:
            result["tool_call_id"] = archive.tool_call_id
        if metadata.get("additional_kwargs"):
            result["additional_kwargs"] = metadata["additional_kwargs"]
        if metadata.get("response_metadata"):
            result["response_metadata"] = metadata["response_metadata"]
        if metadata.get("name"):
            result["name"] = metadata["name"]

        return result


# Singleton instance
message_archive_service = MessageArchiveService()
