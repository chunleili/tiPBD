import subprocess

def test_cloth_amg_python():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=0", "-solver_type=AMG", "-end_frame=2"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_cloth_amg_cuda():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=1", "-solver_type=AMG", "-end_frame=2"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_cloth_xpbd_gpu():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=0", "-solver_type=XPBD","-arch=gpu", "-end_frame=2"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_cloth_xpbd_cpu():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=0", "-solver_type=XPBD","-arch=gpu", "-end_frame=2"]
    ret = subprocess.check_call(run_args)
    assert ret == 0


def test_cloth_direct():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64","-solver_type=DIRECT", "-end_frame=2", "-maxiter=2"]
    ret = subprocess.check_call(run_args)
    assert ret == 0



def test_cloth_xpbd_gpu_calc_strain():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-solver_type=XPBD","-arch=gpu", "-end_frame=2", "-calc_strain=1", "-export_strain=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0


def test_cloth_xpbd_gpu_calc_energy():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-solver_type=XPBD","-arch=gpu", "-end_frame=2", "-calc_energy=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0


def test_cloth_amg_cuda_calc_strain():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=1", "-solver_type=AMG", "-end_frame=2", "-export_strain=1", "-calc_strain=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0

def test_cloth_amg_cuda_calc_energy():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64", "-use_cuda=1", "-solver_type=AMG", "-end_frame=2", "-calc_energy=1"]
    ret = subprocess.check_call(run_args)
    assert ret == 0