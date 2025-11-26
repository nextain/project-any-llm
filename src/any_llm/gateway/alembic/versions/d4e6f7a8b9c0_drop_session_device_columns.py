"""Drop device_type/device_id from session_tokens."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "d4e6f7a8b9c0"
down_revision = "b7e5c8d1f0ab"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_index(op.f("ix_session_tokens_device_type"), table_name="session_tokens")
    op.drop_column("session_tokens", "device_type")
    op.drop_column("session_tokens", "device_id")


def downgrade():
    op.add_column("session_tokens", sa.Column("device_id", sa.String(length=128), nullable=True))
    op.add_column("session_tokens", sa.Column("device_type", sa.String(length=32), nullable=True))
    op.create_index(op.f("ix_session_tokens_device_type"), "session_tokens", ["device_type"], unique=False)
