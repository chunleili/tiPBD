import subprocess

def test_cloth_amg_python():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=0", "-solver_type=AMG", "-end_frame=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_cloth_amg_cuda():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=1", "-solver_type=AMG", "-end_frame=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_cloth_xpbd_gpu():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=0", "-solver_type=XPBD","-arch=gpu", "-end_frame=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_cloth_xpbd_cpu():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=0", "-solver_type=XPBD","-arch=gpu", "-end_frame=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0


def test_cloth_direct():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64" "-solver_type=DIRECT", "-end_frame=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0