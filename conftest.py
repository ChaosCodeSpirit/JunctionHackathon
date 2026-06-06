"""Pytest configuration: add the project root to sys.path.

This lets test files import from the project modules (surface_code,
extract_syndromes, internal_helpers, etc.) without requiring the project
to be installed as a package.
"""
import sys
from pathlib import Path

# Add the project root to sys.path so tests can import project modules
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
