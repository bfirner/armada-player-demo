import argparse
import random
import sys

from dice import ArmadaDice


# Seed with time or a local source of randomness
random.seed()


parser = argparse.ArgumentParser(description='Process dice counts.')
parser.add_argument('--red', type=int, help='number of red dice', default=0)
parser.add_argument('--blue', type=int, help='number of blue dice', default=0)
parser.add_argument('--black', type=int, help='number of black dice', default=0)

args = parser.parse_args()
print("Simulating with {} red, {} blue, {} black".format(args.red, args.blue, args.black))
colors = []
rolls = []
for _ in range(0, args.red):
    colors.append("red")
    roll = ArmadaDice.random_roll("red")
    rolls.append(roll)
for _ in range(0, args.blue):
    colors.append("blue")
    roll = ArmadaDice.random_roll("blue")
    rolls.append(roll)
for _ in range(0, args.black):
    colors.append("black")
    roll = ArmadaDice.random_roll("black")
    rolls.append(roll)

while True:
    for i in range(0, len(colors)):
        print("{}: {} {}".format(i, colors[i], rolls[i]))
    print("reroll or [enter] to quit")
    line = sys.stdin.readline()
    line = line.rstrip()
    if 0 == len(line):
        break;
    rolls[int(line)] = ArmadaDice.random_roll(colors[int(line)])

