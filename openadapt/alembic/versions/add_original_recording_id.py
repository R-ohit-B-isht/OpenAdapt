"""add_original_recording_id.py

Revision ID: add_original_recording_id
Revises:
Create Date: 2024-06-18 04:38:41.171566

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_original_recording_id'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Add the 'original_recording_id' column to the 'recording' table
    op.add_column('recording', sa.Column('original_recording_id', sa.Integer(), nullable=True))

def downgrade():
    # Remove the 'original_recording_id' column from the 'recording' table
    op.drop_column('recording', 'original_recording_id')
