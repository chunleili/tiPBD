import sys, os, argparse

parser = argparse.ArgumentParser()

parser.add_argument("-er", "--log_energy_range", type=int, nargs="+", default=[])
print(parser.parse_args().log_energy_range)
