import sys, os
from pathlib import Path

libdir = Path(__file__).resolve().parents[2]
libdir = libdir / "lib" 
sys.path.append(str(libdir))

import pymgpbd as mp # type: ignore

NV=4
NCONS=2
sim = mp.SolveSoft()
sim.resize_fields(NV, NCONS)
print(sim.pos)