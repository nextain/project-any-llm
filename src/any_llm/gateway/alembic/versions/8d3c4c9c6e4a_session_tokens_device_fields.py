"""Add device fields to session_tokens and drop api_key_id."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "8d3c4c9c6e4a"
down_revision = "3f2f5d9d9b77"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("session_tokens", sa.Column("device_type", sa.String(length=32), nullable=True))
    op.add_column("session_tokens", sa.Column("device_id", sa.String(length=128), nullable=True))
    op.create_index(op.f("ix_session_tokens_device_type"), "session_tokens", ["device_type"], unique=False)

def downgrade():
    op.add_column(
        "session_tokens",
        sa.Column("api_key_id", sa.VARCHAR(), autoincrement=False, nullable=True),
    )
    op.create_foreign_key(
        "session_tokens_api_key_id_fkey",
        "session_tokens",
        "api_keys",
        ["api_key_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_index(op.f("ix_session_tokens_device_type"), table_name="session_tokens")
    op.drop_column("session_tokens", "device_id")
    op.drop_column("session_tokens", "device_type")
