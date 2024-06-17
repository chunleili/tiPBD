def mkdir_if_not_exist(path=None):
    import os
    from pathlib import Path
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory_path):
        os.makedirs(path)