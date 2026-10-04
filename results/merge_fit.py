"""Replace one protocol's entries in paper_fit.json with those from a separate fit.

    python results/merge_fit.py results/fit_hdt.json hdt
"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src, protocol = Path(sys.argv[1]).resolve(), sys.argv[2]
main = json.loads((root / "paper_fit.json").read_text())
extra = json.loads(src.read_text())
main["fits"] = [f for f in main["fits"] if f["protocol"] != protocol] + [
    f for f in extra["fits"] if f["protocol"] == protocol
]
main["settings"].setdefault("merged", {})[protocol] = {"from": str(src.relative_to(root)), **extra["settings"]}
(root / "paper_fit.json").write_text(json.dumps(main, indent=2))
print(f"merged {protocol} from {src}")
