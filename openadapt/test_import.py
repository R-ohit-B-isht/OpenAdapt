import sys
import os

# Ensure the project root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Diagnostic print statements
print("Current sys.path:", sys.path)
print("Current working directory:", os.getcwd())

# Attempt to import the openadapt module
try:
    from openadapt.config import config
    print("Successfully imported openadapt module.")
except ModuleNotFoundError as e:
    print("Error importing openadapt module:", e)
