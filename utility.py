#
# Copyright Bernhard Firner, 2019-2020
#
# Utility functions.

import csv
import math
import torch

from game_constants import (ArmadaDimensions, ArmadaTypes)
from ship import Ship

# Array of matching token indexes
def token_index(token, defender):
    return [idx for idx in range(len(defender.defense_tokens)) if token in defender.defense_tokens[idx]]


def get_rotation_matrix(heading):
    """Get the rotation matrix for the given heading.

    Arguments:
        heading (float): Rotation in radians.
    Returns:
        torch.tensor: 2x2 rotation matrix
    """
    return torch.tensor([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])


def get_corners(ship):
    """Get the corner coordinates of a ship.

    This is a useful function for collisions, distances, etc. Traversing the returned edges in
    order will walk the perimeter of the object.

    Arguments:
        ship (Ship): The ship.
    Returns:
        torch.tensor: 4x2 matrix with the x,y coordinates of the corners in the order: left rear,
                      left front, right rear, right front.
    """
    # Get the ship location information
    location = ship.get_range('location')
    heading = ship.get('heading')
    size = ArmadaDimensions.ship_bases_feet[ArmadaDimensions.size_names[int(ship.get('size'))]]
    # To find the corners first rotate and then add in the current location translation
    corners = torch.zeros(4, 2)
    for xi, x in enumerate([-1, 1]):
        # We flip the order of iteration through the y dimension based upon the x so that the
        # corners end up in order. This is helpful for future iterations. The iteration sequence is
        # [-1,1], [-1,-1], [1,-1], [1, 1].
        for yi, y in enumerate([x * -1, x * 1]):
            corners[2*xi + yi, 0] = x * size[0]/2.
            corners[2*xi + yi, 1] = y * size[1]/2.
    # Now rotate. Translation is done next so this rotation is about the origin.
    rotation_matrix = get_rotation_matrix(heading)
    # TODO FIXME Corners are not being rotated about the correct reference point and the rectangle
    # is being turned into a trapezoid
    corners = torch.matmul(corners, rotation_matrix)
    # Now translate
    corners[:,0] += location[0]
    corners[:,1] += location[1]
    return corners


def get_edges(corners):
    """Get the equations for the edges (the perimeter) of an object given its corners.

    This is a useful function for collissions, distances, etc for objects defined by line segments.
    The corners must be ordered so that a walk from index 0, 1, ..., -1, 0 defines the perimeter of
    the object.

    Arguments:
        ship (Ship): The ship.
    Returns:
        (List[tuple(float, float, float)]): List of tuples of (x origin, x end, y origin, slope)
    """
    location = ship.get_range('location')
    heading = ship.get_range('heading')
    edges = []
    with torch.no_grad:
        perimeter = torch.cat((corners, corners[:1]))
        for i in range(corners.size(0)):
            first = perimeter[i].item()
            second = perimeter[i + 1].item()
            slope = ((second[1] - first[1]) / (second[0] - first[0])).item()
            edges.append(first, second, perimeted[i][1].item(), slope)
    return edges


def find_intersection(line_a, line_b):
    """Find the intersection point (if it exists) of two line segments, a and b.

    This uses an algorithm based upon LeMothe's "Tricks of the Windows Game Programming Gurus". We
    could have used some fancy cross product stuff here, but this is much more comprehensible.
    Each segment is parameterized by variables, s and t, such at when s=0 we are at line_a[0] and
    when s[1] we are at line_a[1] (and so on for t). We then solve for the values of s and t where
    the equations for line_a and line_b give the same result. If we cannot find values for s and t
    that satisfy this then there is no intersection.

    Arguments:
        line_a (tensor): 2x2 tensor describing line a. line_a[0] is the first point.
        line_b (tensor): The same thing is line_a.
    Returns:
        None or 2 element torch.tensor
    """
    xslope_a = line_a[1,0] - line_a[0,0]
    xslope_b = line_b[1,0] - line_b[0,0]
    yslope_a = line_a[1,1] - line_a[0,1]
    yslope_b = line_b[1,1] - line_b[0,1]
    # Find the parameters where the lines intersect
    s_numerator = -yslope_a * (line_a[0,0] - line_b[0,0]) + xslope_a * (line_a[0,1] - line_b[0, 1])
    t_numerator = -yslope_b * (line_a[0,0] - line_b[0, 0]) + xslope_b * (line_a[0,1] - line_b[0,1])
    denominator = (-xslope_b * yslope_a + xslope_a * yslope_b)
    # If the lines are parallel then the slope will cancel out and the denominator will be 0.
    # For simplicity we will just say that they do not intersect in that case.
    if 0 == denominator:
        # Early return for efficiency
        return None
    # Check one range at a time to possibly skip a second division operation
    s = s_numerator / denominator
    if 0 <= s and s <= 1:
        t = t_numerator / denominator
        if 0 <= t and t <= 1:
            intersection = [line_a[0,0] + t * xslope_a, line_a[0, 1] + t * yslope_a]
            return torch.tensor(intersection)
    # No intersection
    return None


# TODO Write functions to measure distances and ranges between objects
def ruler_distance(a, b):
    """Return the distance between two objects

    Distance is measured as an index into ArmadaTypes.ruler_distance_feet or a number outside of the
    outside to indicate being outside of the ruler.

    Arguments:
        a (object): Measure distance from this object
        b (object): Measure distance to this object
    Returns:
        tuple(int, (float, float)): Index into ArmadaTypes.ruler_distance_feet or an index outside
                                    of the table if the distance is greater than the ruler length
                                    and the coordinates of the shortest path.
    Raises:
        ValueError if a or b are not recognized types.
    """
    if not (isinstance(a, Ship) and isinstance(b, Ship)):
        raise ValueError("Cannot measure distances between non-Ship types.")
    # First check for an overlap. Do this by seeing if any lines overlap. This can be done by taking
    # two endpoint of an edge for ship A (ax1, ay1), (ax2, ay2) and two endpoint of an edge of ship
    # B (bx1, by1), (bx2, by2) and doing some cross product stuff. If the lines intersect then the
    # angle from (ax1, ay1), (ax2, ay2), (bx1, by1) and (ax1, ay1), (ax2, ay2), (bx2, by2) must be
    # different because the endpoints of B's line must line on either side of A's line because the
    # lines cut their one another. This should also hold true going from B's line to the endpoints
    # of A's line since they must cut each other. If not, then the line segment would cut the other
    # if it were longer, but it falls short.
    a_corners = get_corners(a)
    b_corners = get_corners(b)
    # Simplify some things by repeating the first point so we go through every edge segment.
    a_perimeter = torch.cat((a_corners, a_corners[:1]))
    b_perimeter = torch.cat((b_corners, b_corners[:1]))
    # Brute force looping.
    for ai in range(a_corners.size(0)):
        for bi in range(b_corners.size(0)):
            a_segment = a_perimeter[ai:ai+2]
            b_segment = b_perimeter[bi:bi+2]
            intersect = find_intersection(a_segment, b_segment)
            if intersect is not None:
                return 0, (intersect, intersect)

    # No intersection, we will have to find the closest point.
    # First notice that if we only draw lines from the corners of objects 'a' and 'b' then this is
    # sufficient. If we find the shortest path from each corner to the side of the other object we
    # end up with 32 lines.
    # There are two cases for each shortest line:
    # 1. We can draw a line from the corner that is perpendicular to the side. This is the shortest
    # path.
    # 2. The perpendicular line does not intersect the side. In this case one of the corners is
    # closest.
    # Obviously this can be optimized to skip the 2 farther sides. With 3 corners and 2 sides on
    # each ship we would only search 12 possible shortest lines, but we would need to do additional
    # comparisons to determine the closest sides. Since this is a discussion of constant time none
    # of this will have a large impact so this function will be as simple as possible, unlike these
    # comments.

    # Calculate the corner distances
    distance = torch.nn.PairwiseDistance()
    # Future proofing this code a bit by checking the size of the corners rather than assuming 4.
    a_num_corners = a_corners.size(0)
    b_num_corners = b_corners.size(0)
    distances = torch.zeros(a_num_corners * b_num_corners)
    for corner in range(a_corners.size(0)):
        out_index = corner * a_num_corners
        distances[out_index:out_index + b_num_corners] = distance(
            a_corners[corner].expand(b_num_corners, 2), b_corners)
    # Also need to calculate the perpendicular line distances
    # To simplify things we can just calculate the distance from the two sides adjacent to the two
    # closest points
    min_corner_distance, index = torch.min(distances, 0)
    a_corner = int(math.floor(index.item() / b_num_corners))
    b_corner = int((index.item() % b_num_corners))
    # Now check the distances from the corners to the sides. If a perpendicular line from the side
    # of one ship cannot be drawn to the corner of the other then the corner line is shorter.
    shortest_path = min_corner_distance
    shortest_points = (a_corners[a_corner], b_corners[b_corner])
    # First find the distance from a_corner to the sides of b and b_corner to the sides of a
    possible_paths = []
    for offset in range(b_corner - 1, b_corner + 1):
        possible_paths.append((a_corners[a_corner],
                               (b_corners[offset], b_corners[(offset + 1) % b_num_corners])))
    for offset in range(a_corner - 1, a_corner + 1):
        possible_paths.append((b_corners[b_corner],
                               (a_corners[offset], a_corners[(offset + 1) % a_num_corners])))
    # TODO Calculate the perpendicular distance from the side to the corner. If a perpendicular line
    # cannot be drawn then the corner to corner distance is the best possible.
    for corner, side in possible_paths:
        # Calculate the line perpendicular to the side could go from the side to the corner
        side_slope = (side[1][1] - side[0][1]) / (side[1][0] - side[0][0])
        # We'll skip being clever here and just make a gigantic perpendicular line and then check
        # for an intersection.
        if 0. == side_slope:
            perp_point_one = torch.tensor([corner[0].item(),
                                           corner[1].item() - ArmadaDimensions.play_area_height_feet])
            perp_point_two = torch.tensor([corner[0].item(),
                                           corner[1].item() + ArmadaDimensions.play_area_height_feet])
        else:
            perp_slope = 1. / side_slope
            perp_point_one = torch.tensor([corner[0].item() - ArmadaDimensions.play_area_width_feet,
                                           corner[1].item() - ArmadaDimensions.play_area_width_feet * perp_slope])
            perp_point_two = torch.tensor([corner[0].item() + ArmadaDimensions.play_area_width_feet,
                                           corner[1].item() + ArmadaDimensions.play_area_width_feet * perp_slope])
        intercept = find_intersection(torch.stack(side), torch.stack((perp_point_one, perp_point_two)))
        if intercept is not None:
            distance = math.sqrt((corner[0].item() - intercept[0].item())**2 +
                                 (corner[1].item() - intercept[1].item())**2)
            if distance < shortest_path:
                shortest_path = distance
                shortest_points = (corner, intercept)
    # Find the ruler index for the shortest past
    ruler_index = 0
    ruler = ArmadaDimensions.ruler_distance_feet
    while ruler_index < len(ruler) and shortest_path > ruler[ruler_index]:
        ruler_index += 1
    return ruler_index, shortest_points


# TODO Write a function to output the world state in a visualizable format


def tokens_available(token, defender, accuracy_tokens = None):
    """Return a tuple indicating if a red or green token is available.

    Arguments:
        token (str)   : The token types (one of ArmadaTypes.defense_tokens)
        defender(Ship): The defending ship whose tokens to check.
    Returns:
        tuple(bool, bool): True if a green or red token is available, respectively.
    """
    green = False
    red = False
    token_offset = ArmadaTypes.defense_tokens.index(token)
    green_offset, green_size = defender.get_index("green_defense_tokens")
    red_offset, red_size = defender.get_index("red_defense_tokens")
    green_offset += token_offset
    red_offset += token_offset
    green_sum = defender.encoding[green_offset].item()
    red_sum = defender.encoding[green_offset].item()
    if accuracy_tokens:
        green_sum = max(0., green_sum - accuracy_tokens[token_offset])
        red_sum -= max(0., red_sum - accuracy_tokens[token_offset + len(ArmadaTypes.defense_tokens)])
    return (green_sum, red_sum)

def max_damage_index(pool_faces):
    for face in ["hit_crit", "hit_hit", "crit", "hit"]:
        if face in pool_faces:
            return pool_faces.index(face)
    return None

def face_index(face, pool_faces):
    return [idx for idx in range(len(pool_faces)) if face in pool_faces[idx]]

def parseShips(filename):
    keys = {}
    ship_templates = {}
    with open(filename, newline='') as ships:
        shipreader = csv.reader(ships, delimiter=',', quotechar='|')
        rowcount = 0
        for row in shipreader:
            # parse the header first to find the column keys
            if ( 0 == rowcount ):
                count = 0
                for key in row:
                    count = count + 1
                    keys[count] = key
            else:
                newship = {}
                count = 0
                # Fill in all of the information on this vessel
                for key in row:
                    count = count + 1
                    newship[keys[count]] = key
                # Create a new ship template
                ship_templates[newship['Ship Name']] = newship
            rowcount = rowcount + 1
    return keys, ship_templates

def print_roll(colors, roll):
    for i in range(0, len(colors)):
        print("{}: {} {}".format(i, colors[i], roll[i]))
