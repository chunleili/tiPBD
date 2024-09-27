# clean ./latest* and ./case* directories in the current directory

import os
import shutil

def clean():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(this_dir)

    for f in os.listdir("."):
        if f.startswith("latest") or f.startswith("case"):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)

    # clean meta/std_err*.txt 
    for f in os.listdir("meta"):
        if f.startswith("stderr"):
            os.remove(os.path.join("meta", f))
        

if __name__ == "__main__":
    clean()