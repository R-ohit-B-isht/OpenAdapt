import os
import sys

# Ensure the project root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the PYTHONPATH environment variable
os.environ['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Diagnostic print statements
print("Current sys.path:", sys.path)
print("Current PYTHONPATH:", os.environ['PYTHONPATH'])

# Run Alembic migration
from alembic.config import Config
from alembic import command

alembic_cfg = Config("/home/ubuntu/OpenAdapt/openadapt/alembic.ini")
command.upgrade(alembic_cfg, "head")
