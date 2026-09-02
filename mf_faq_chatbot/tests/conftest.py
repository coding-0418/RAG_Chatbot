import sys
from pathlib import Path

# app.py / rag.py / prompts.py use bare imports (not a package), so make the
# project directory importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
