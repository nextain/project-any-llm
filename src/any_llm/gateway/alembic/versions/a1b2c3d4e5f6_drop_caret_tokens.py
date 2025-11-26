"""Drop refresh_token and access_token_expires_at from caret_users."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "8d3c4c9c6e4a"
branch_labels = None
depends_on = None


def upgrade():
    pass

def downgrade():
    pass
