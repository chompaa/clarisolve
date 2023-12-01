from .iccnn import ICCNN
from .icres import ICRes
from .srcnn import SRCNN
from .srcnnc import SRCNNC
from .srres import SRRes

SR_MODELS = {
    "srcnn": SRCNN,
    "srcnnc": SRCNNC,
    "srres": SRRes,
}

IC_MODELS = {
    "iccnn": ICCNN,
    "icres": ICRes,
}
