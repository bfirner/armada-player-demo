import math
import PyGnuplot as gp

from game_constants import ArmadaDimensions
from utility import get_corners, parseShips, ruler_distance
from ship import Ship

keys, ship_templates = parseShips('data/test_ships.csv')
# Make two ships
alice = Ship(name="Alice", template=ship_templates["Attacker"], upgrades=[], player_number=1)
bob = Ship(name="Bob", template=ship_templates["Attacker"], upgrades=[], player_number=2)

# Put them somewhere
alice.set(name="location", value=[1.5, 1.5])
alice.set(name="heading", value=math.pi/2.)
print(f"Alice location is {alice.get_range('location')}")
bob.set(name="location", value=[1.5, 0.8])
bob.set(name="heading", value=math.pi/4.)
print(f"Bob location is {bob.get_range('location')}")

# Get the distance
distance, path = ruler_distance(alice, bob)

# Print out the ship edges and the shortest path
print(f"Distance is {distance}")
print(f"Path is {path}")

# Plot with gnuplot
print("set size ratio 0.5")
print(f"set xrange [0:{ArmadaDimensions.play_area_width_feet}]")
print(f"set yrange [0:{ArmadaDimensions.play_area_height_feet}]")
def print_ship(corners, name):
    print(f"set label '{name}' at {corners[0,0]}, {corners[0,1]} font',8'")
    print(f"set arrow from {corners[1,0]}, {corners[1,1]} to {corners[0,0]}, {corners[0,1]}")
    for idx in range(1,4):
        print(f"set arrow from {corners[idx,0]}, {corners[idx,1]} to {corners[(idx+1)%4,0]}, {corners[(idx+1)%4,1]} nohead")

print_ship(get_corners(alice), "Alice")
print_ship(get_corners(bob), "Bob")

print(f"set arrow from {path[0][0]}, {path[0][1]} to {path[1][0]}, {path[1][1]} heads")
print("plot -1 notitle")

#gp.default_term = 'x11'
#gp.figure(number=None, term='x11')
gp.c("set size ratio 0.5")
gp.c(f"set xrange [0:{ArmadaDimensions.play_area_width_feet}]")
gp.c(f"set yrange [0:{ArmadaDimensions.play_area_height_feet}]")
def plot_ship(corners, name):
    gp.c(f"set label '{name}' at {corners[0,0]}, {corners[0,1]} font',8'")
    gp.c(f"set arrow from {corners[1,0]}, {corners[1,1]} to {corners[0,0]}, {corners[0,1]}")
    for idx in range(1,4):
        gp.c(f"set arrow from {corners[idx,0]}, {corners[idx,1]} to {corners[(idx+1)%4,0]}, {corners[(idx+1)%4,1]} nohead")

plot_ship(get_corners(alice), "Alice")
plot_ship(get_corners(bob), "Bob")

gp.c(f"set arrow from {path[0][0]}, {path[0][1]} to {path[1][0]}, {path[1][1]} heads")
gp.c("plot -1 notitle")
#gp.figure()
