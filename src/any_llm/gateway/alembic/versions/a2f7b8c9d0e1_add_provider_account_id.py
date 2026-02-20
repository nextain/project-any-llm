"""Add provider_account_id to caret_users.

Revision ID: a2f7b8c9d0e1
Revises: 6a9c1d2e3f4b
Create Date: 2026-02-21 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a2f7b8c9d0e1"
down_revision: str | Sequence[str] | None = "6a9c1d2e3f4b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("caret_users", sa.Column("provider_account_id", sa.String(), nullable=True))
    op.create_index("ix_caret_users_provider_account_id", "caret_users", ["provider_account_id"])


def downgrade() -> None:
    op.drop_index("ix_caret_users_provider_account_id", table_name="caret_users")
    op.drop_column("caret_users", "provider_account_id")
