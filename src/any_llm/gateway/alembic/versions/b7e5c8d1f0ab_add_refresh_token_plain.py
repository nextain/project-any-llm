"""Add refresh_token_plain to session_tokens."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "b7e5c8d1f0ab"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("session_tokens", sa.Column("refresh_token_plain", sa.String(), nullable=True))


def downgrade():
    op.drop_column("session_tokens", "refresh_token_plain")
