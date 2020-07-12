# Ship class and supporting classes

from collections import OrderedDict
from enum import Enum
import torch

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
    def __init__(self, name, template, upgrades, player_number, device=None):
        """Contsruct a specific instance of a ship.
        
        Args:
            name (str)                   : Name for this vessel.
            template (ShipType)          : Ship template to copy.
            upgrades (table str->str)    : Upgrades to equip.
            player_number (int)          : The player who controls this ship.
            device (str)                 : Default Tensor type ('cuda' or 'cpu'). Automatic if None.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoding = torch.zeros(Ship.encodeSize()).to(device)

        self.encoding.fill_(0.)
        # Initialize attributes of this specific ship instance
        self.set('player', player_number)
        self.set('hull', int(template["Hull"]))

        self.set("ship", 0.)
        self.set("size", ArmadaDimensions.size_names.index(template['Size'].lower()))
        idx, length = Ship.get_index("defense_tokens")
        self.encoding[idx:idx + length] = 0.
        for ttype in ArmadaTypes.defense_tokens:
            tname = "Defense Token {}".format(ttype.capitalize())
            token_idx = idx + ArmadaTypes.defense_tokens.index(ttype)
            if tname in template:
                self.encoding[token_idx] = int(template[tname])

        # Max shields (current shields will be filled in the reset function)
        idx = Ship.get_index("max_shields")[0]
        for zone in ['left', 'right', 'front', 'rear']:
            name = "Shields {}".format(zone.capitalize())
            self.encoding[idx + ArmadaTypes.hull_zones.index(zone)] = int(template[name])
        if 'Huge' == template['Size']:
            for zone in ['left-auxiliary', 'right-auxiliary']:
                name = "Shields {} {}".format(zone.capitalize())
                self.encoding[idx + ArmadaTypes.hull_zones.index(zone)] = int(template[name])

        # Presence of hull zones/firing arcs
        idx, length = Ship.get_index("hull_zones")
        self.encoding[idx:idx + length] = 0.
        # Set the hull zones to indicate which are present
        idx = Ship.get_index("hull_zones")[0]
        for zone in ['left', 'right', 'front', 'rear']:
            self.encoding[idx + ArmadaTypes.hull_zones.index(zone)] = 1.
        if 'Huge' == template['Size']:
            for zone in ['left-auxiliary', 'right-auxiliary']:
                self.encoding[idx + ArmadaTypes.hull_zones.index(zone)] = 1.

        # Initialize the armaments
        idx = Ship.get_index("dice")[0]
        for i, zone in enumerate(['left', 'right', 'front', 'rear']):
            for j, color in enumerate(ArmadaDice.die_colors):
                name = "Armament {} {}".format(zone.capitalize(), color.capitalize())
                hull_offset = ArmadaTypes.hull_zones.index(zone)
                if 0 < len(template[name]):
                    number = int(template[name])
                else:
                    number = 0
                self.encoding[idx + hull_offset * len(ArmadaDice.die_colors) + j] = number
        
        if 'Huge' == template['Size']:
            for i, zone in enumerate(['left-auxiliary', 'right-auxiliary']):
                for j, color in enumerate(ArmadaDice.die_colors):
                    name = "Armament {} {}".format(zone.capitalize(), color.capitalize())
                    hull_offset = ArmadaTypes.hull_zones.index(zone)
                    number = int(template[name])
                    self.encoding[idx + hull_offset * len(ArmadaDice.die_colors) + j] = number

        self.name = name

        # TODO Check for legality and actually handle
        self.upgrades = upgrades
        self.width, self.height = ArmadaDimensions.ship_bases[template['Size'].lower()]
        self.reset()

    @staticmethod
    def _initialize_encoding():
        """Initialize the _enc_index and _enc_len variables."""
        Ship._enc_index = OrderedDict()
        Ship._enc_len = OrderedDict()
        Ship._enc_index['player'] = 0
        Ship._enc_len['player'] = 1
        Ship._enc_index['hull'] = 1
        Ship._enc_len['hull'] = 1
        Ship._enc_index['damage'] = 2
        Ship._enc_len['damage'] = 1
        # TODO Face up damage card effects
        Ship._enc_index['speed'] = 3
        Ship._enc_len['speed'] = 1
        Ship._enc_index['ship'] = 4
        Ship._enc_len['ship'] = 1
        # Encode the size as a single value since different sizes are linearly related
        Ship._enc_index['size'] = 5
        Ship._enc_len['size'] = 1
        # Defense tokens and state belong here, whether the token has been spend during this
        # attack step is stored in the attack state
        # TODO FIXME Make this a loop starting with cur_idx:=6 and use a table of names and their
        # lengths, this is too error prone.
        Ship._enc_index['defense_tokens'] = 6
        Ship._enc_len['defense_tokens'] = len(ArmadaTypes.defense_tokens)
        cur_idx = Ship._enc_index['defense_tokens'] + Ship._enc_len['defense_tokens']

        Ship._enc_index['green_defense_tokens'] = cur_idx
        Ship._enc_len['green_defense_tokens'] = len(ArmadaTypes.defense_tokens)
        cur_idx = Ship._enc_index['green_defense_tokens'] + Ship._enc_len['green_defense_tokens']

        Ship._enc_index['red_defense_tokens'] = cur_idx
        Ship._enc_len['red_defense_tokens'] = len(ArmadaTypes.defense_tokens)
        cur_idx = Ship._enc_index['red_defense_tokens'] + Ship._enc_len['red_defense_tokens']

        Ship._enc_index['max_shields'] = cur_idx
        Ship._enc_len['max_shields'] = len(ArmadaTypes.hull_zones)
        cur_idx = Ship._enc_index['max_shields'] + Ship._enc_len['max_shields']

        Ship._enc_index['shields'] = cur_idx
        Ship._enc_len['shields'] = len(ArmadaTypes.hull_zones)
        cur_idx = Ship._enc_index['shields'] + Ship._enc_len['shields']

        # Presence of particular hull zones
        Ship._enc_index['hull_zones'] = cur_idx
        Ship._enc_len['hull_zones'] = len(ArmadaTypes.hull_zones)
        cur_idx = Ship._enc_index['hull_zones'] + Ship._enc_len['hull_zones']
        # Armament for each zone
        Ship._enc_index['dice'] = cur_idx
        Ship._enc_len['dice'] = len(ArmadaTypes.hull_zones) * len(ArmadaDice.die_colors)
        cur_idx = Ship._enc_index['dice'] + Ship._enc_len['dice']
        # TODO Line of sight marker locations and firing arc locations
        # TODO Upgrades
        # TODO Ignition arc
        Ship._enc_index['commands'] = cur_idx
        Ship._enc_len['commands'] = ArmadaTypes.max_command_dials
        # Location is a pair of x and y coordinates in feet (since that is the range ruler size).
        Ship._enc_index['location'] = cur_idx
        Ship._enc_len['location'] = 2
        cur_idx = Ship._enc_index['location'] + Ship._enc_len['location']
        # The heading is the rotation of the ship
        Ship._enc_index['heading'] = cur_idx
        Ship._enc_len['heading'] = 1

    @staticmethod
    def encodeSize():
        """Get the size of the ship encoding.

        Returns:
            int: Size of the ship encoding (number of Tensor elements)
        """
        # Programmatically initialize the index lookup if it doesn't exist
        if not hasattr(Ship, '_enc_index'):
            Ship._initialize_encoding()
        last_key = list(Ship._enc_index.keys())[-1]
        size = Ship._enc_index[last_key] + Ship._enc_len[last_key]
        return size

    @staticmethod
    def get_index(data_name):
        """Get the index of a data element.

        Arguments:
            data_name(str): Name of the data element.
        Returns:
            (int, int): Tuple of the beginning of the data and the length.
        """
        # Programmatically initialize the index lookup if it doesn't exist
        if not hasattr(Ship, '_enc_index'):
            Ship._initialize_encoding()

        if data_name not in Ship._enc_index:
            raise RuntimeError("Ship has no attribute named {}".format(data_name))
        return (Ship._enc_index[data_name], Ship._enc_len[data_name])

    def base_size(self):
        """Get the ship width and length.

        Returns:
            tuple(int, int): width and length
        """
        index = self.encoding[Ship._enc_index['size']]
        return ArmadaDimensions.ship_bases[ArmadaDimensions.size_names[index]]

    def token_count(self, index):
        """Get the number of green and red tokens at the given index.

        The index corresponds to a particular type of token as defined in
        ArmadaTypes.defense_tokens.

        Returns:
            tuple(int, int): The number of green and red tokens.
        """
        green_idx = Ship._enc_index["green_defense_tokens"]
        red_idx = Ship._enc_index["red_defense_tokens"]
        return self.encoding[green_idx + index], self.encoding[red_idx + index]

    def ready_defense_tokens(self):
        """Replace all red tokens with green versions."""
        with torch.no_grad():
            # Add the red tokens to the green tokens and set red tokens to 0
            green_idx = Ship._enc_index["green_defense_tokens"]
            red_idx = Ship._enc_index["red_defense_tokens"]
            token_len = Ship._enc_len['green_defense_tokens']
            self.encoding[green_idx:green_idx + token_len] += self.encoding[red_idx:red_idx + token_len]
            self.encoding[red_idx:red_idx + src_len] = 0.

    def spend_token(self, token_type, color_type):
        """Spend a token of the given type and color.

        Args:
            token_type (str): Token type to spend.
            color_type (int): 0 for green, 1 for red
        """
        red_idx = Ship._enc_index["red_defense_tokens"]
        type_offset = ArmadaTypes.defense_tokens.index(token_type)
        if 0 == color_type:
            green_idx = Ship._enc_index["green_defense_tokens"]
            self.encoding[green_idx + type_offset] -= 1
            self.encoding[red_idx + type_offset] += 1
        else:
            self.encoding[red_idx + type_offset] -= 1


    def ready_upgrade_cards(self):
        """Unexhaust upgrade cards."""
        # Not implemented yet
        pass

    def adjacent_zones(self, zone):
        """Return hull zones adjacent to the given zone."""
        index = int(self.encoding[Ship._enc_index['size']].item())
        size = ArmadaDimensions.size_names[index]
        if size == 'huge':
            if zone not in ArmadaTypes.adjacent_huge_hull_zones:
                raise RuntimeError("Unrecognized hull zone {}".format(zone))
            return ArmadaTypes.adjacent_huge_hull_zones[zone]
        else:
            if zone not in ArmadaTypes.adjacent_hull_zones:
                raise RuntimeError("Unrecognized hull zone {}".format(zone))
            return ArmadaTypes.adjacent_hull_zones[zone]


    def get(self, name):
        """Get a value from the encoding.
        
        Arguments:
            name  (str): Name of the encoding field.
        Returns:
            value (float): The value of the encoding with the given name.
        """
        index, length = Ship.get_index(name)
        if 1 == length:
            return self.encoding[index].item()
        else:
            raise RuntimeError("Use Ship.get_range for single element data.")


    def get_range(self, name):
        """Get a view of the encoding of a field with multiple elements.
        
        Arguments:
            name  (str): Name of the encoding field.
        Returns:
            value (torch.Tensor): The tensor is a view of the original data, clone or convert to a
                                  list to avoid modification.
        """
        index, length = Ship.get_index(name)
        if 1 == length:
            raise RuntimeError("Use Ship.get for single element data.")
        else:
            return self.encoding[index:index + length]


    def set(self, name, value):
        """Set a value in encoding.
        
        Arguments:
            name  (str): Name of the encoding field.
            value (numeric, List, or torch.Tensor): A value assignable to a tensor.
        """
        vtype = type(value)
        if vtype is not int and vtype is not float and vtype is not list and vtype is not torch.Tensor:
            raise RuntimeError('Ship.set does not have data type "{}"'.format(vtype))
        index, length = Ship.get_index(name)
        if 1 == length:
            self.encoding[index] = value
        else:
            if type(value) is int or type(value) is float:
                raise RuntimeError("Attempt to assign a scalar value to an encoding range.")
            # Convert a list to a tensor to assign a range
            if type(value) is list:
                self.encoding[index:index + length] = torch.tensor(value)
            else:
                self.encoding[index:index + length] = value


    def set_range(self, name, value):
        """Set a range in the encoding to a value.

        Arguments:
            name      (str): Name of the encoding field.
            value (numeric): Value to set.
        """
        vtype = type(value)
        if vtype is not int and vtype is not float:
            raise RuntimeError('Ship.set_range does not support data type "{}"'.format(vtype))
        index, length = Ship.get_index(name)
        self.encoding[index:index + length] = value


    def reset(self):
        """Resets shields, hull, and defense tokens and initialize values in the encoding."""
        self.set("damage", 0.)
        self.set("speed", 0.)
        self.set_range("commands", 0.)

        # Set defense tokens, and shields
        # Initialize all tokens as green
        self.set('green_defense_tokens', self.get_range('defense_tokens'))
        self.set_range('red_defense_tokens', 0.)

        self.set('shields', self.get_range('max_shields'))

        # Set a location off of the board. Lump each player's ships together.
        self.set("location", [-1., self.get('player') * -1.])
        self.set("heading", 0.)

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
        # TODO Extreme range
        # Roll red dice at all valid ranges
        die_offset = Ship._enc_index['dice']
        hull_offset = die_offset + ArmadaTypes.hull_zones.index(zone) * len(ArmadaDice.die_colors)
        if distance in ["short", "medium", "long"]:
            red_offset = ArmadaDice.die_colors.index("red")
            num_dice = int(self.encoding[hull_offset + red_offset].item())
            colors = colors + ["red"] * num_dice
        # Roll blue dice at all short to medium
        if distance in ["short", "medium"]:
            blue_offset = ArmadaDice.die_colors.index("blue")
            num_dice = int(self.encoding[hull_offset + blue_offset].item())
            colors = colors + ["blue"] * num_dice
        # Roll black dice at short range
        if distance in ["short"]:
            black_offset = ArmadaDice.die_colors.index("black")
            num_dice = int(self.encoding[hull_offset + black_offset].item())
            colors = colors + ["black"] * num_dice
        # TODO FIXME Only gathering should happen in the ship, rolling should follow in a different
        # area of code
        for color in colors:
            faces.append(ArmadaDice.random_roll(color))
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
        offset = Ship._enc_index['shields'] + ArmadaTypes.hull_zones.index(zone)
        shields = int(self.encoding[offset].item())
        if shields >= amount:
            shields -= amount
            self.encoding[offset] = shields
            return 0
        else:
            amount -= shields
            self.encoding[offset] = 0.
            return amount

    def damage(self, zone, amount):
        """
        Deal damage to a hull zone.

        Args:
            zone (str): One of ArmadaTypes.hull_zones or "hull"
            amount (int): Amount of damage
        """
        damage_offset = Ship._enc_index['damage']
        damage = int(self.encoding[damage_offset].item())
        if "hull" == zone:
            damage += amount
        else:
            shield_offset = Ship._enc_index['shields'] + ArmadaTypes.hull_zones.index(zone)
            shields = int(self.encoding[shield_offset].item())
            if shields >= amount:
                shields -= amount
            else:
                amount -= shields
                shields = 0
                damage += amount
            self.encoding[shield_offset] = shields
        self.encoding[damage_offset] = damage

    def hull(self):
        hull_offset = Ship._enc_index['hull']
        hull = int(self.encoding[hull_offset].item())
        return hull

    def damage_cards(self):
        damage_offset = Ship._enc_index['damage']
        damage = int(self.encoding[damage_offset].item())
        return damage

    def stringify(self):
        """Return a string version of the ship."""
        shield_offset = Ship._enc_index['shields']
        shield_length = Ship._enc_len['shields']
        shields = self.encoding[shield_offset:shield_offset + shield_length]
        green_def_idx = Ship._enc_index['green_defense_tokens']
        green_def_len = Ship._enc_len['green_defense_tokens']
        green_tokens = self.encoding[green_def_idx:green_def_idx + green_def_len]
        red_def_idx = Ship._enc_index['red_defense_tokens']
        red_def_len = Ship._enc_len['red_defense_tokens']
        red_tokens = self.encoding[red_def_idx:red_def_idx + red_def_len]
        return str(
            "{}: hull ({}/{}), shields {}, green defense tokens {}, red defense tokens {}".format(
                self.name, self.damage_cards(), self.hull(), shields, green_tokens, red_tokens))

    def __str__(self):
        return self.stringify()

    def __repr__(self):
        return self.stringify()

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
