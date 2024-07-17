from subprocess import call
import os

def config():
    call(["cmake", "-B", "build"])

def build():
    call(["cmake", "--build", "build", "--config", "Release"])

os.chdir(os.path.dirname((__file__)))
config()
build()