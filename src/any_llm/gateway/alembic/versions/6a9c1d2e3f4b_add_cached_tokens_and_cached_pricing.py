"""Add cached_tokens to usage_logs and cached_price_per_million to model_pricing."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6a9c1d2e3f4b"
down_revision: str | Sequence[str] | None = "f0f1e2d3c4b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    usage_logs_cols = {col["name"] for col in inspector.get_columns("usage_logs")}
    if "cached_tokens" not in usage_logs_cols:
        op.add_column("usage_logs", sa.Column("cached_tokens", sa.Integer(), nullable=True))

    model_pricing_cols = {col["name"] for col in inspector.get_columns("model_pricing")}
    if "cached_price_per_million" not in model_pricing_cols:
        op.add_column("model_pricing", sa.Column("cached_price_per_million", sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    usage_logs_cols = {col["name"] for col in inspector.get_columns("usage_logs")}
    if "cached_tokens" in usage_logs_cols:
        op.drop_column("usage_logs", "cached_tokens")

    model_pricing_cols = {col["name"] for col in inspector.get_columns("model_pricing")}
    if "cached_price_per_million" in model_pricing_cols:
        op.drop_column("model_pricing", "cached_price_per_million")
