import os
import sys

# Ensure the project root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the PYTHONPATH environment variable
os.environ['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Diagnostic print statements
print("Current sys.path:", sys.path)
print("Current PYTHONPATH:", os.environ['PYTHONPATH'])

# Test import of openadapt module
try:
    from openadapt.config import config
    print("Successfully imported openadapt module.")
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
