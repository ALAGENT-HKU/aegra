"""Fix message_archive dedup: unique on (thread_id, message_id) instead of (thread_id, message_index)

The original unique constraint on (thread_id, message_index) causes silent message loss
under concurrent archive calls because max_index is computed at read time with a race window.
This migration switches deduplication to (thread_id, message_id) which is the message's
immutable identity assigned by LangGraph.

Revision ID: 20260129_fix_archive_idx
Revises: 20260128_message_archive
Create Date: 2026-03-31

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260129_fix_archive_idx"
down_revision = "20260128_message_archive"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Replace unique (thread_id, message_index) with partial unique (thread_id, message_id)."""

    # 1. Drop old unique index on (thread_id, message_index)
    op.drop_index("idx_message_archive_thread_index", table_name="message_archive")

    # 2. Recreate as non-unique (still needed for ORDER BY performance)
    op.create_index(
        "idx_message_archive_thread_index",
        "message_archive",
        ["thread_id", "message_index"],
    )

    # 3. Add partial unique index on (thread_id, message_id) WHERE message_id IS NOT NULL
    op.execute(
        """
        CREATE UNIQUE INDEX idx_message_archive_thread_msg_id
        ON message_archive (thread_id, message_id)
        WHERE message_id IS NOT NULL
        """
    )


def downgrade() -> None:
    """Revert to unique (thread_id, message_index)."""

    op.drop_index("idx_message_archive_thread_msg_id", table_name="message_archive")
    op.drop_index("idx_message_archive_thread_index", table_name="message_archive")
    op.create_index(
        "idx_message_archive_thread_index",
        "message_archive",
        ["thread_id", "message_index"],
        unique=True,
    )
