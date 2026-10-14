"""Run from a source checkout without installing: python scripts/esdl.py --help."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "src"))

from esdl.cli import main

raise SystemExit(main())
