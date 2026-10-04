#!/usr/bin/env python3
"""Replace one protocol's fits in fit.json with those from another fit file.

    python scripts/merge_fit.py results/fit_hdt.json hdt
"""

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src, protocol = Path(sys.argv[1]).resolve(), sys.argv[2]
fits = json.loads((root / "fit.json").read_text())
extra = json.loads(src.read_text())
fits["fits"] = [f for f in fits["fits"] if f["protocol"] != protocol] + [
    f for f in extra["fits"] if f["protocol"] == protocol
]
fits["settings"].setdefault("merged", {})[protocol] = {"from": str(src.relative_to(root)), **extra["settings"]}
(root / "fit.json").write_text(json.dumps(fits, indent=2))
print(f"merged {protocol} from {src}")
