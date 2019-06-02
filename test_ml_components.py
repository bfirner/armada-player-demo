# Should be run with pytest:
# > python3 -m pytest

import numpy
import pytest
import torch

import ship
import utility
from game_engine import handleAttack
from learning_agent import (LearningAgent)
from world_state import (WorldState)
from game_constants import (ArmadaTypes)

# Initialize ships from the test ship list
keys, ship_templates = utility.parseShips('data/test_ships.csv')

# Test that the world state encoding is giving expected results

def make_encoding(ship_a, ship_b, attack_range, agent):
    """This function calculates the average time to destruction when a shoots at b.

    Args:
      ship_a ((Ship, str)): Attacker and hull zone tuple.
      ship_b ((Ship, str)): Defender and hull zone tuple.
      trials (int): Number of trials in average calculation.
      range (str): Attack range.
    
    """
    roll_counts = []
    # Reset ship b for each trial
    world_state = WorldState()
    world_state.addShip(ship_a, 0)
    world_state.addShip(ship_b, 1)

    pool_colors, pool_faces = ship_a.roll("front", attack_range)
    attack_dict = {
        'range': attack_range,
        'attacker': ship_a,
        'attacking zone': "front",
        'defender': ship_b,
        'defending zone': "front",
        'pool_colors': pool_colors,
        'pool_faces': pool_faces,
        # Keep track of which tokens are spent.
        # Tokens cannot be spent multiple times in a single attack.
        'spent_tokens': {},
        # A token targeted with an accuracy cannot be spent.
        'accuracy_tokens': []
    }
    world_state.updateAttack(attack_dict)

    return agent.encodeAttackState(world_state), world_state


def test_token_encodings():
    """Test that the encoding is correct for different defense tokens."""

    agent = LearningAgent()

    no_token = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)
    one_brace = ship.Ship(name="Single Brace", template=ship_templates["Single Brace"], upgrades=[], player_number=2)
    two_brace = ship.Ship(name="Double Brace", template=ship_templates["Double Brace"], upgrades=[], player_number=2)
    two_redirect = ship.Ship(name="Double Redirect", template=ship_templates["Double Redirect"], upgrades=[], player_number=2)
    two_evade = ship.Ship(name="Double Evade", template=ship_templates["Double Evade"], upgrades=[], player_number=2)
    two_contain = ship.Ship(name="Double Contain", template=ship_templates["Double Contain"], upgrades=[], player_number=2)
    two_scatter = ship.Ship(name="Double Scatter", template=ship_templates["Double Scatter"], upgrades=[], player_number=2)

    enc_one_brace = make_encoding(no_token, one_brace, "short", agent)[0][0]
    enc_two_brace = make_encoding(one_brace, two_brace, "short", agent)[0][0]
    enc_two_redirect = make_encoding(two_brace, two_redirect, "short", agent)[0][0]
    enc_two_evade = make_encoding(two_redirect, two_evade, "short", agent)[0][0]
    enc_two_contain = make_encoding(two_evade, two_contain, "short", agent)[0][0]
    enc_two_scatter = make_encoding(two_contain, two_scatter, "short", agent)[0][0]

    # Check the token section of the encoding. It occurs after the two hull and shield sections.
    # Skip the spent section as well
    token_types = len(ArmadaTypes.defense_tokens)
    token_begin = 2 * 5 + token_types
    hot_token_size = 8

    def get_tokens(t, color):
        # There are six slots of [type - 5, color - 2, accuracy targeted - 1]
        # We need to check the type and color in this function
        tokens = torch.tensor([0.0, 0, 0, 0, 0])
        for slot in range(ArmadaTypes.max_defense_tokens):
            offset = token_begin + hot_token_size * slot
            #print("Checking slot")
            #print(t[offset:offset + token_types + len(ArmadaTypes.token_colors)])
            if 1.0 == t[offset + token_types + ArmadaTypes.token_colors.index(color)]:
                tokens = tokens + t[offset:offset + token_types]
        return tokens
    
    # Order of tokens in the encoding
    # token_types = ["evade", "brace", "scatter", "contain", "redirect"]
    # Check the green token section
    assert torch.equal(get_tokens(enc_one_brace, "green"), (torch.tensor([0, 1.0, 0, 0, 0])))
    assert torch.equal(get_tokens(enc_two_brace, "green"), (torch.tensor([0, 2.0, 0, 0, 0])))
    assert torch.equal(get_tokens(enc_two_redirect, "green"), (torch.tensor([0, 0, 0, 0, 2.0])))
    assert torch.equal(get_tokens(enc_two_evade, "green"), (torch.tensor([2.0, 0, 0, 0, 0])))
    assert torch.equal(get_tokens(enc_two_contain, "green"), (torch.tensor([0, 0, 0, 2.0, 0])))
    assert torch.equal(get_tokens(enc_two_scatter, "green"), (torch.tensor([0, 0, 2.0, 0, 0])))


def test_red_token_encodings():
    """Test that the encoding is correct for red defense tokens."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    two_brace = ship.Ship(name="Double Brace", template=ship_templates["Double Brace"], upgrades=[], player_number=2)
    two_redirect = ship.Ship(name="Double Redirect", template=ship_templates["Double Redirect"], upgrades=[], player_number=2)
    two_evade = ship.Ship(name="Double Evade", template=ship_templates["Double Evade"], upgrades=[], player_number=2)
    two_contain = ship.Ship(name="Double Contain", template=ship_templates["Double Contain"], upgrades=[], player_number=2)
    two_scatter = ship.Ship(name="Double Scatter", template=ship_templates["Double Scatter"], upgrades=[], player_number=2)

    two_brace.defense_tokens.append("red brace")
    two_brace.defense_tokens.append("red brace")
    two_redirect.defense_tokens.append("red redirect")
    two_redirect.defense_tokens.append("red redirect")
    two_evade.defense_tokens.append("red evade")
    two_evade.defense_tokens.append("red evade")
    two_contain.defense_tokens.append("red contain")
    two_contain.defense_tokens.append("red contain")
    two_scatter.defense_tokens.append("red scatter")
    two_scatter.defense_tokens.append("red scatter")

    enc_two_brace = make_encoding(attacker, two_brace, "short", agent)[0][0]
    enc_two_redirect = make_encoding(attacker, two_redirect, "short", agent)[0][0]
    enc_two_evade = make_encoding(attacker, two_evade, "short", agent)[0][0]
    enc_two_contain = make_encoding(attacker, two_contain, "short", agent)[0][0]
    enc_two_scatter = make_encoding(attacker, two_scatter, "short", agent)[0][0]

    # Check the token section of the encoding. It occurs after the two hull and shield sections.
    # Skip the spent section as well
    token_types = len(ArmadaTypes.defense_tokens)
    token_begin = 2 * 5 + token_types
    hot_token_size = 8

    def get_tokens(t, color):
        # There are six slots of [type - 5, color - 2, accuracy targeted - 1]
        # We need to check the type and color in this function
        tokens = torch.tensor([0.0, 0, 0, 0, 0])
        for slot in range(ArmadaTypes.max_defense_tokens):
            offset = token_begin + hot_token_size * slot
            print("Checking slot")
            print(t[offset:offset + token_types + len(ArmadaTypes.token_colors)])
            if 1.0 == t[offset + token_types + ArmadaTypes.token_colors.index(color)]:
                tokens = tokens + t[offset:offset + token_types]
        return tokens
    
    # Order of tokens in the encoding
    # token_types = ["evade", "brace", "scatter", "contain", "redirect"]
    # Check the red token section
    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("brace")] = 2.0
    assert torch.equal(get_tokens(enc_two_brace, "red"), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("redirect")] = 2.0
    assert torch.equal(get_tokens(enc_two_redirect, "red"), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("evade")] = 2.0
    assert torch.equal(get_tokens(enc_two_evade, "red"), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("contain")] = 2.0
    assert torch.equal(get_tokens(enc_two_contain, "red"), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("scatter")] = 2.0
    assert torch.equal(get_tokens(enc_two_scatter, "red"), ttensor)


def test_spent_encodings():
    """Test that the encoding is correct for different defense tokens."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    two_brace = ship.Ship(name="Double Brace", template=ship_templates["Double Brace"], upgrades=[], player_number=2)
    two_redirect = ship.Ship(name="Double Redirect", template=ship_templates["Double Redirect"], upgrades=[], player_number=2)
    two_evade = ship.Ship(name="Double Evade", template=ship_templates["Double Evade"], upgrades=[], player_number=2)
    two_contain = ship.Ship(name="Double Contain", template=ship_templates["Double Contain"], upgrades=[], player_number=2)
    two_scatter = ship.Ship(name="Double Scatter", template=ship_templates["Double Scatter"], upgrades=[], player_number=2)

    two_brace.defense_tokens.append("red brace")

    # Modify an attack state to indicate that tokens were spent
    #attack_dict = {
    #    'range': attack_range,
    #    'attacker': attacker[0],
    #    'attacking zone': attacker[1],
    #    'defender': defender[0],
    #    'defending zone': defender[1],
    #    'pool_colors': pool_colors,
    #    'pool_faces': pool_faces,
    #    # Keep track of which tokens are spent.
    #    # Tokens cannot be spent multiple times in a single attack.
    #    'spent_tokens': {},
    #    # A token targeted with an accuracy cannot be spent.
    #    'accuracy_tokens': []
    #}
    #world_state.updateAttack(attack_dict)

    enc_two_brace, world_state = make_encoding(attacker, two_brace, "short", agent)
    enc_two_brace = enc_two_brace[0]

    # Check the token section of the encoding. It occurs after the two hull and shield sections.
    # Skip the spent section as well
    token_types = len(ArmadaTypes.defense_tokens)
    token_begin = 2 * 5 + token_types
    hot_token_size = 8

    def get_tokens(t, color):
        # There are six slots of [type - 5, color - 2, accuracy targeted - 1]
        # We need to check the type and color in this function
        tokens = torch.tensor([0.0, 0, 0, 0, 0])
        for slot in range(ArmadaTypes.max_defense_tokens):
            offset = token_begin + hot_token_size * slot
            #print("Checking slot")
            #print(t[offset:offset + token_types + len(ArmadaTypes.token_colors)])
            if 1.0 == t[offset + token_types + ArmadaTypes.token_colors.index(color)]:
                tokens = tokens + t[offset:offset + token_types]
        return tokens
    
    # Order of tokens in the encoding
    # token_types = ["evade", "brace", "scatter", "contain", "redirect"]
    # Check the token section
    assert torch.equal(get_tokens(enc_two_brace, "green"), (torch.tensor([0, 2.0, 0, 0, 0])))

    # Verify that no tokens are marked spent by default
    # token_types = ["evade", "brace", "scatter", "contain", "redirect"]
    # world_state.attack["spend_tokens"]

    # Check the token section of the encoding

    # Spend a token and verify that a token is not marked as spent
    spent_state = agent.encodeAttackState(world_state)


def test_range_encodings():
    """Test that the encoding is correct for ranges."""

    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    no_token = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)
    for attack_range in ['long', 'medium', 'short']:
        pass


def test_roll_encodings():
    """Test that the encoding is correct for dice pools and faces."""

    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    no_token = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)
