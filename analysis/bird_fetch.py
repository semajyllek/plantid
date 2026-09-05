"""Fetch the pre-registered Oregon bird sets (BIRDS_PREREG.md)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import safety_fetch as sf

sets = json.load(open(Path(__file__).parent / "bird_sets.json"))
sf.SPECIES = sets["crowded"] + sets["varied"]
sf.OUT = Path(__file__).parent / "birds"
if __name__ == "__main__":
    sf.main()
