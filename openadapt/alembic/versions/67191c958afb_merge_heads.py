"""Merge heads

Revision ID: 67191c958afb
Revises: add_original_recording_id, bb25e889ad71
Create Date: 2024-06-21 18:42:22.692546

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '67191c958afb'
down_revision = ('add_original_recording_id', 'bb25e889ad71')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
