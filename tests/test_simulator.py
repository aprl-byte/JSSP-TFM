import numpy as np
from jsspc.instances import generate, Instance
from jsspc.simulator import run, CLASSIC, validate

FT06 = [[(2,1),(0,3),(1,6),(3,7),(5,3),(4,6)],[(1,8),(2,5),(4,10),(5,10),(0,10),(3,4)],
        [(2,5),(3,4),(5,8),(0,9),(1,1),(4,7)],[(1,5),(0,5),(2,5),(3,3),(4,8),(5,9)],
        [(2,9),(1,3),(4,5),(5,4),(0,3),(3,1)],[(1,3),(3,3),(5,9),(0,10),(4,4),(2,1)]]

def ft06():
    return Instance("ft06", np.array([[a for a,_ in j] for j in FT06]), np.array([[b for _,b in j] for j in FT06]))

def test_ft06_pilot_values():
    # valores del experimento preliminar (Tabla 1 de la primera entrega)
    exp = {"SPT": 94, "LPT": 86, "MWKR": 67, "LWKR": 96, "MOPNR": 60, "FIFO_est": 68}
    inst = ft06()
    for r, v in exp.items():
        assert run(inst, CLASSIC[r]).makespan == v, r

def test_schedules_feasible():
    for seed in range(5):
        inst = generate(7, 5, seed)
        for r, f in CLASSIC.items():
            res = run(inst, f)
            assert validate(inst, res.start)
            assert res.makespan >= inst.lower_bound()
