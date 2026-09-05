"""Twenty common Oregon plants, family-diverse — a normally-composed user list.

The safety set was chosen to be maximally confusable and is not what a person's
list looks like. This is: twenty plants someone in Portland would actually want
named, spanning twenty genera and a dozen families.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import safety_fetch as sf

sf.SPECIES = [
    "Pseudotsuga menziesii", "Gaultheria shallon", "Berberis aquifolium",
    "Polystichum munitum", "Acer circinatum", "Arbutus menziesii",
    "Thuja plicata", "Tsuga heterophylla", "Trillium ovatum", "Camassia quamash",
    "Achillea millefolium", "Oxalis oregana", "Symphoricarpos albus",
    "Holodiscus discolor", "Physocarpus capitatus", "Cornus sericea",
    "Lonicera involucrata", "Ribes sanguineum", "Salix scouleriana",
    "Quercus garryana",
]
sf.OUT = Path(__file__).parent / "common"

if __name__ == "__main__":
    sf.main()
