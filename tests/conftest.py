"""Make `hydrogen` importable when the package isn't installed.

If you `pip install -e .` first, this is a no-op. Otherwise we just prepend the project
root to `sys.path` so `import hydrogen` resolves directly from the source tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
