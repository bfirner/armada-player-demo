# Ship class and supporting classes

from enum import Enum

from dice import ArmadaDice
from game_constants import (
    ArmadaDimensions,
    ArmadaTypes
)

class UpgradeType(Enum):
    commander             = 1
    officer               = 2
    weapons_team          = 3
    support_team          = 4
    offensive_retrofit    = 5
    defensive_retrofit    = 6
    turbolasers           = 7
    ion_cannons           = 8
    ordnance              = 9
    fleet_support         = 10
    experimental_retrofit = 11
    boarding_team         = 12
    title                 = 13

class Armament:
    def __init__(self, redCount, blueCount, blackCount):
        self.red   = redCount
        self.blue  = blueCount
        self.black = blackCount

class ShipType:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = attributes

class Ship:
    def __init__(self, name, template, upgrades, player_number):
        """Contsruct a specific instance of a ship.
        
        Args:
            name (str)                   : Name for this vessel.
            template (ShipType)          : Ship template to copy.
            upgrades (table str->str)    : Upgrades to equip.
            player_number (int)          : The player who controls this ship.
        """
        self.name = name
        self.player_number = player_number
        # Deep copy the attributes from the template
        self.attributes = {}
        for attr in template:
            key = attr.lower()
            self.attributes[key] = template[attr]
            if 'armament' in key or 'defense token' in key:
                if 0 == len(self.attributes[key]):
                    self.attributes[key] = 0
                else:
                    self.attributes[key] = int(self.attributes[key])
        # TODO Check for legality
        self.upgrades = upgrades
        self.width, self.height = ArmadaDimensions.ship_bases[template['Size'].lower()]
        self.reset()

    def token_type(self, index):
        """Get the string name of the token type in the given index.

        Arguments:
            index (int): Index to check
        Returns:
            str: Name of the token type.
        """
        name = self.defense_tokens[index]
        if 'red' == name[:len('red')]:
            return name[len('red '):]
        else:
            return name[len('green '):]

    def ready_defense_tokens(self):
        """Replace all red tokens with green versions."""
        for token in self.defense_tokens:
            if 'red' in token:
                token = 'green' + token[3:]

    def ready_upgrade_cards(self):
        """Unexhaust upgrade cards."""
        # Not implemented yet
        pass

    def adjacent_zones(self, zone):
        """Return hull zones adjacent to the given zone."""
        if self.attributes['size'] == 'Huge':
            if zone not in ArmadaTypes.adjacent_huge_hull_zones:
                raise RuntimeError("Unrecognized hull zone {}".format(zone))
            return ArmadaTypes.adjacent_huge_hull_zones[zone]
        else:
            if zone not in ArmadaTypes.adjacent_hull_zones:
                raise RuntimeError("Unrecognized hull zone {}".format(zone))
            return ArmadaTypes.adjacent_hull_zones[zone]

    def reset(self):
        """Resets shields, hull, and defense tokens."""
        # Initialize attributes of this specific ship instance
        self._hull = int(self.attributes["hull"])
        self.command_dials = []
        # Initialize shields and defence tokens
        self.shields = {}
        self.shields["left"] = int(self.attributes["shields left"])
        self.shields["right"] = int(self.attributes["shields right"])
        self.shields["front"] = int(self.attributes["shields front"])
        self.shields["rear"] = int(self.attributes["shields rear"])
        if self.attributes['size'] == 'Huge':
            self.shields["left-auxiliary"] = int(self.attributes["shields left auxiliary"])
            self.shields["right-auxiliary"] = int(self.attributes["shields right auxiliary"])
        self.defense_tokens = []
        for token in ArmadaTypes.defense_tokens:
            str_token = "defense token " + token
            # Insert a green token for each available token
            if str_token in self.attributes:
                for _ in range(self.attributes[str_token]):
                    self.defense_tokens.append("green " + token)
        # Tokens spent in a spend defense tokens phase
        self.spent_tokens =  [False] * len(self.defense_tokens)

    def leave_spend_defense_tokens(self):
        """No tokens should be in the spent state outside of this phase."""
        self.spent_tokens =  [False] * len(self.defense_tokens)

    def roll(self, zone, distance):
        """
        return an attack roll for the given arc at the given range.

        Args:
            zone (str) : One of front, left, right, and rear
            distance (str) : short, medium, or long
        Returns an array of colors and faces
        """
        colors = []
        faces = []
        # Roll red dice at all valid ranges
        if distance in ["short", "medium", "long"]:
            for _ in range(0, self.attributes["armament "+ zone + " red"]):
                colors.append("red")
                faces.append(ArmadaDice.random_roll("red"))
        # Roll blue dice at all short to medium
        if distance in ["short", "medium"]:
            for _ in range(0, self.attributes["armament "+ zone + " blue"]):
                colors.append("blue")
                faces.append(ArmadaDice.random_roll("blue"))
        # Roll black dice at short range
        if distance == "short":
            for _ in range(0, self.attributes["armament "+ zone + " black"]):
                colors.append("black")
                faces.append(ArmadaDice.random_roll("black"))
        return colors, faces

    def shield_damage(self, zone, amount):
        """
        Deal damage to a hull zone but only deplete the shields, don't assign hull damage. Return
        the amount of damage that will be assigned to the hull.

        Args:
            zone (str): One of ArmadaTypes.hull_zones
            amount (int): Amount of damage
        Returns:
            (int): Amount of damage that will be assigned to the hull.
        """
        if self.shields[zone] >= amount:
            self.shields[zone] -= amount
            return 0
        else:
            amount -= self.shields[zone]
            self.shields[zone] = 0
            return amount

    def damage(self, zone, amount):
        """
        Deal damage to a hull zone.

        Args:
            zone (str): One of ArmadaTypes.hull_zones or "hull"
            amount (int): Amount of damage
        """
        if "hull" == zone:
            self._hull -= amount
        elif self.shields[zone] >= amount:
            self.shields[zone] -= amount
        else:
            amount -= self.shields[zone]
            self.shields[zone] = 0
            self._hull -= amount
        # Hull cannot be negative.
        if 0 > self._hull:
            self._hull = 0

    def hull(self):
        return self._hull

    def __str__(self):
        return str("{}: hull ({}), shields {}, defense tokens ({})".format(self.name, self.hull(), self.shields, self.defense_tokens))

    def __repr__(self):
        return str("{}: hull ({}), shields {}, defense tokens ({})".format(self.name, self.hull(), self.shields, self.defense_tokens))
               

def parseShips(filename):
    """ Returns a list of ships."""
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
    ship_types = {}
    for name, attributes in ship_templates.items():
        ship_types[name] = ShipType(name, attributes)
        #print("{}:".format(name))
        #for a_name, a_value in attributes.items():
        #    print("    {} : {}".format(a_name, a_value))
    return ship_types
