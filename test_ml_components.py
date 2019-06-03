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
        if 1.0 == t[offset + token_types + ArmadaTypes.token_colors.index(color)]:
            tokens = tokens + t[offset:offset + token_types]
    return tokens

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
    
    # Order of tokens in the encoding
    # token_types = ["evade", "brace", "scatter", "contain", "redirect"]
    # Check the green token section
    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("brace")] = 1.0
    assert torch.equal(get_tokens(enc_one_brace, "green"), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("brace")] = 2.0
    assert torch.equal(get_tokens(enc_two_brace, "green"), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("redirect")] = 2.0
    assert torch.equal(get_tokens(enc_two_redirect, "green"), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("evade")] = 2.0
    assert torch.equal(get_tokens(enc_two_evade, "green"), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("contain")] = 2.0
    assert torch.equal(get_tokens(enc_two_contain, "green"), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("scatter")] = 2.0
    assert torch.equal(get_tokens(enc_two_scatter, "green"), ttensor)


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

    enc_two_brace, world_state = make_encoding(attacker, two_brace, "short", agent)
    enc_two_brace = enc_two_brace[0]
    
    # Order of tokens in the encoding
    # token_types = ["evade", "brace", "scatter", "contain", "redirect"]
    # Check the token section
    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("brace")] = 2.0
    assert torch.equal(get_tokens(enc_two_brace, "green"), ttensor)

    # Verify that no tokens are marked spent by default
    spent_begin = 2 * 5
    assert torch.sum(enc_two_brace[spent_begin:spent_begin + token_types]) == 0


    # Try spending some different token types
    for ttype in ArmadaTypes.defense_tokens:
        spent_section = {}
        spent_section[ttype] = True
        world_state.attack['spent_tokens'] = spent_section
        spent_enc = agent.encodeAttackState(world_state)[0]
        ttensor = torch.tensor([0.0, 0, 0, 0, 0])
        ttensor[ArmadaTypes.defense_tokens.index(ttype)] = 1.0
        assert torch.equal(spent_enc[spent_begin:spent_begin + token_types], ttensor)


def test_range_encodings():
    """Test that the encoding is correct for ranges."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[],
                         player_number=1)
    no_token = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"],
                         upgrades=[], player_number=2)

    range_begin = 2 * 5 + hot_token_size * ArmadaTypes.max_defense_tokens + token_types
    for offset, attack_range in enumerate(ArmadaTypes.ranges):
        enc_attack = make_encoding(attacker, no_token, attack_range, agent)[0][0]
        assert torch.sum(enc_attack[range_begin:range_begin + len(ArmadaTypes.ranges)]) == 1
        assert 1.0 == enc_attack[range_begin + offset].item()


def test_roll_encodings():
    """Test that the encoding is correct for dice pools and faces."""

    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    no_token = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)
