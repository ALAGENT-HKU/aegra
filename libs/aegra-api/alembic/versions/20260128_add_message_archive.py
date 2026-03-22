"""Add message_archive table for full history preservation

This migration creates the message_archive table to store complete message history
that survives SummarizationMiddleware compression in LangGraph checkpoints.

Revision ID: 20260128_add_message_archive
Revises: 9fc3b51cedf7
Create Date: 2026-01-28

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260128_message_archive"
down_revision = "d042a0ca1cb5"  # Last migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create message_archive table for full history preservation."""
    
    op.create_table(
        "message_archive",
        sa.Column(
            "id",
            sa.Text(),
            server_default=sa.text("uuid_generate_v4()::text"),
            nullable=False,
        ),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("message_index", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=True),
        sa.Column("message_type", sa.Text(), nullable=False),
        sa.Column(
            "content",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "tool_calls",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("tool_call_id", sa.Text(), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["thread.thread_id"],
            ondelete="CASCADE",
        ),
    )
    
    # Create indexes
    op.create_index(
        "idx_message_archive_thread",
        "message_archive",
        ["thread_id"],
    )
    op.create_index(
        "idx_message_archive_thread_index",
        "message_archive",
        ["thread_id", "message_index"],
        unique=True,
    )


def downgrade() -> None:
    """Drop message_archive table."""
    op.drop_index("idx_message_archive_thread_index", table_name="message_archive")
    op.drop_index("idx_message_archive_thread", table_name="message_archive")
    op.drop_table("message_archive")
