import subprocess

def test_soft_amg_python():
    run_args = ["python", "engine/soft/soft3d.py", "-use_cuda=0", "-solver_type=AMG", "-end_frame=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_soft_amg_cuda():
    run_args = ["python", "engine/soft/soft3d.py", "-use_cuda=1", "-solver_type=AMG", "-end_frame=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_soft_xpbd_gpu():
    run_args = ["python", "engine/soft/soft3d.py", "-use_cuda=0", "-solver_type=XPBD","-arch=gpu", "-end_frame=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_soft_xpbd_cpu():
    run_args = ["python", "engine/soft/soft3d.py", "-use_cuda=0", "-solver_type=XPBD","-arch=gpu", "-end_frame=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0


def test_soft_direct():
    run_args = ["python", "engine/soft/soft3d.py", "-solver_type=DIRECT", "-end_frame=1", "-maxiter=2"]
    ret = subprocess.check_call(run_args)
    assert ret == 0