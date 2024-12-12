
import pymgpbd as mp

NV=2
NCONS=1
sim = mp.SolveSoft()
sim.resize_fields(NV, NCONS)
print(sim.pos)