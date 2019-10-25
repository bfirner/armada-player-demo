# Should be run with pytest:
# > python3 -m pytest

import numpy
import pytest
import torch

import ship
import utility
from armada_encodings import (Encodings)
from dice import (ArmadaDice)
from learning_agent import (LearningAgent)
from world_state import (AttackState, WorldState)
from game_constants import (ArmadaTypes)

# Initialize ships from the test ship list
keys, ship_templates = utility.parseShips('data/test_ships.csv')

# Check the token section of the encoding. It occurs after the two hull and shield sections.
# Skip the spent section as well
token_types = len(ArmadaTypes.defense_tokens)

def get_tokens(t, color, hot_token_size):
    """
    Returns:
        torch.Tensor: A tensor with numbers of tokens of the given color. The index into the tensor
                      corresponds to a token type in the ArmadaTypes.defense_tokens list.
    """
    # There are six slots of [type - 5, color - 2, accuracy targeted - 1]
    # We need to check the type and color in this function
    # The defense token spent section begins after the defender's shields and hull
    token_begin = 1 + len(ArmadaTypes.hull_zones)
    token_types = len(ArmadaTypes.defense_tokens)
    found_tokens = torch.tensor([0.0] * token_types)
    for slot in range(ArmadaTypes.max_defense_tokens):
        offset = token_begin + hot_token_size * slot
        # If this is the correct color then add this token to the count
        if 1.0 == t[offset + token_types + ArmadaTypes.token_colors.index(color)]:
            found_tokens = found_tokens + t[offset:offset + token_types]
    print("Found tokens {}".format(found_tokens))
    return found_tokens

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
    attack = AttackState(attack_range=attack_range, attacker=ship_a, attacking_hull="front",
        defender=ship_b, defending_hull="front", pool_colors=pool_colors, pool_faces=pool_faces)
    world_state.updateAttack(attack)

    # The defense token and die locations have been reordered in the encoding, put them back to
    # their original ordering here.
    encoding, defense_token_mapping, die_slot_mapping = Encodings.encodeAttackState(world_state)
    Encodings.inPlaceUnmap(encoding, defense_token_mapping, die_slot_mapping)

    return encoding, world_state


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
    assert torch.equal(get_tokens(enc_one_brace, "green", Encodings.hot_token_size), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("brace")] = 2.0
    assert torch.equal(get_tokens(enc_two_brace, "green", Encodings.hot_token_size), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("redirect")] = 2.0
    assert torch.equal(get_tokens(enc_two_redirect, "green", Encodings.hot_token_size), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("evade")] = 2.0
    assert torch.equal(get_tokens(enc_two_evade, "green", Encodings.hot_token_size), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("contain")] = 2.0
    assert torch.equal(get_tokens(enc_two_contain, "green", Encodings.hot_token_size), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("scatter")] = 2.0
    assert torch.equal(get_tokens(enc_two_scatter, "green", Encodings.hot_token_size), ttensor)


def test_red_token_encodings():
    """Test that the encoding is correct for red defense tokens."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    two_brace = ship.Ship(name="Double Brace", template=ship_templates["Double Brace"], upgrades=[], player_number=2)
    two_redirect = ship.Ship(name="Double Redirect", template=ship_templates["Double Redirect"], upgrades=[], player_number=2)
    two_evade = ship.Ship(name="Double Evade", template=ship_templates["Double Evade"], upgrades=[], player_number=2)
    two_contain = ship.Ship(name="Double Contain", template=ship_templates["Double Contain"], upgrades=[], player_number=2)
    two_scatter = ship.Ship(name="Double Scatter", template=ship_templates["Double Scatter"], upgrades=[], player_number=2)

    for i, token in enumerate(two_brace.defense_tokens):
        two_brace.defense_tokens[i] = token.replace("green", "red")
    for i, token in enumerate(two_redirect.defense_tokens):
        two_redirect.defense_tokens[i] = token.replace("green", "red")
    for i, token in enumerate(two_evade.defense_tokens):
        two_evade.defense_tokens[i] = token.replace("green", "red")
    for i, token in enumerate(two_contain.defense_tokens):
        two_contain.defense_tokens[i] = token.replace("green", "red")
    for i, token in enumerate(two_scatter.defense_tokens):
        two_scatter.defense_tokens[i] = token.replace("green", "red")

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
    assert torch.equal(get_tokens(enc_two_brace, "red", Encodings.hot_token_size), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("redirect")] = 2.0
    assert torch.equal(get_tokens(enc_two_redirect, "red", Encodings.hot_token_size), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("evade")] = 2.0
    assert torch.equal(get_tokens(enc_two_evade, "red", Encodings.hot_token_size), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("contain")] = 2.0
    assert torch.equal(get_tokens(enc_two_contain, "red", Encodings.hot_token_size), ttensor)

    ttensor = torch.tensor([0.0, 0, 0, 0, 0])
    ttensor[ArmadaTypes.defense_tokens.index("scatter")] = 2.0
    assert torch.equal(get_tokens(enc_two_scatter, "red", Encodings.hot_token_size), ttensor)


def test_spent_encodings():
    """Test that the encoding is correct for different defense tokens."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    defender = ship.Ship(name="Defender", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)

    encoding, world_state = make_encoding(attacker, defender, "short", agent)
    encoding = encoding[0]

    # The defense token spent section begins after the defender's shields and hull
    spent_begin = 1 + len(ArmadaTypes.hull_zones)
    spent_end = spent_begin + ArmadaTypes.max_defense_tokens * Encodings.hot_token_size
    spent_mask = [0] * (len(ArmadaTypes.defense_tokens) + len(ArmadaTypes.token_colors)) + [1]
    spent_mask = torch.Tensor(spent_mask * ArmadaTypes.max_defense_tokens).byte()

    # Verify that no tokens are marked spent by default
    assert torch.sum(encoding[spent_begin:spent_end].masked_select(spent_mask)) == 0

    # TODO FIXME The above is correct now, the below needs to be updated.

    # Spend all of the tokens
    for tidx in range(len(defender.defense_tokens)):
        world_state.attack.defender_spend_token(tidx)

    encoding, defense_token_mapping, die_slot_mapping = Encodings.encodeAttackState(world_state)
    Encodings.inPlaceUnmap(encoding, defense_token_mapping, die_slot_mapping)
    assert torch.sum(encoding[0][spent_begin:spent_end].masked_select(spent_mask)) == len(defender.defense_tokens)

    # Try spending the tokens at different indices
    for tidx in range(len(defender.defense_tokens)):
        # Re-encode and then set the token to spent.
        attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)
        defender = ship.Ship(name="Defender", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)
        encoding, world_state = make_encoding(attacker, defender, "short", agent)
        world_state.attack.defender_spend_token(tidx)
        encoding, defense_token_mapping, die_slot_mapping = Encodings.encodeAttackState(world_state)
        encoding = Encodings.inPlaceUnmap(encoding, defense_token_mapping, die_slot_mapping)
        assert torch.sum(encoding[0][spent_begin:spent_end].masked_select(spent_mask)) == 1.0
        assert encoding[0][spent_begin:spent_end].masked_select(spent_mask)[tidx] == 1.0


def test_accuracy_encodings():
    """Test that the encoding is correct for dice targetted by an accuracy."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    three_brace = ship.Ship(name="Double Brace", template=ship_templates["Triple Brace"], upgrades=[], player_number=2)

    # Make the last token red
    three_brace.defense_tokens[-1] = three_brace.defense_tokens[-1].replace("green", "red")

    enc_three_brace, world_state = make_encoding(attacker, three_brace, "short", agent)

    # Define the offsets for convenience
    token_begin = Encodings.getAttackTokenOffset()
    token_end = token_begin + ArmadaTypes.max_defense_tokens

    green_offset = len(ArmadaTypes.defense_tokens) + ArmadaTypes.token_colors.index("green")
    red_offset = len(ArmadaTypes.defense_tokens) + ArmadaTypes.token_colors.index("red")

    # Verify that no tokens are targeted at first 
    assert 0.0 == enc_three_brace[0, token_begin:token_end].sum()

    # Now target the red token
    world_state.attack.accuracy_tokens[len(three_brace.defense_tokens) - 1] = True
    encoding, dt_mapping, die_mapping = Encodings.encodeAttackState(world_state)
    enc_three_brace = Encodings.inPlaceUnmap(encoding, dt_mapping, die_mapping)[0]

    print("token encoding is {}".format(enc_three_brace[token_begin:token_end]))
    # Verify that only the red token has the accuracy flag set
    for token_idx in range(len(three_brace.defense_tokens)):
        if "green" in three_brace.defense_tokens[token_idx]:
            assert 0.0 == enc_three_brace[token_begin + token_idx].item()
        elif "red" in three_brace.defense_tokens[token_idx]:
            assert 1.0 == enc_three_brace[token_begin + token_idx].item()

    # Target both green tokens
    world_state.attack.accuracy_tokens[0] = True
    world_state.attack.accuracy_tokens[1] = True
    world_state.attack.accuracy_tokens[2] = False
    encoding, dt_mapping, die_mapping = Encodings.encodeAttackState(world_state)
    enc_three_brace = Encodings.inPlaceUnmap(encoding, dt_mapping, die_mapping)[0]

    # Verify that only the green tokens have the accuracy flag set
    for token_slot in range(ArmadaTypes.max_defense_tokens):
        if "green" in three_brace.defense_tokens[token_idx]:
            assert 1.0 == enc_three_brace[token_begin + token_idx].item()
        elif "red" in three_brace.defense_tokens[token_idx]:
            assert 0.0 == enc_three_brace[token_begin + token_idx].item()

    # Target all tokens
    world_state.attack.accuracy_tokens[0] = True
    world_state.attack.accuracy_tokens[1] = True
    world_state.attack.accuracy_tokens[2] = True
    encoding, dt_mapping, die_mapping = Encodings.encodeAttackState(world_state)
    enc_three_brace = Encodings.inPlaceUnmap(encoding, dt_mapping, die_mapping)[0]

    # Verify that all tokens have the accuracy flag set
    assert 3.0 == enc_three_brace[token_begin:token_end].sum()


def test_range_encodings():
    """Test that the encoding is correct for ranges."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[],
                         player_number=1)
    no_token = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"],
                         upgrades=[], player_number=2)

    range_begin = Encodings.getAttackRangeOffset()
    for offset, attack_range in enumerate(ArmadaTypes.ranges):
        enc_attack = make_encoding(attacker, no_token, attack_range, agent)[0][0]
        assert torch.sum(enc_attack[range_begin:range_begin + len(ArmadaTypes.ranges)]) == 1
        assert 1.0 == enc_attack[range_begin + offset].item()


def test_roll_encodings():
    """Test that the encoding is correct for dice pools and faces."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)
    no_token = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)

    dice_begin = Encodings.getAttackDiceOffset()

    # Do 100 trials to ensure everything is working as expected
    _, world_state = make_encoding(attacker, no_token, "short", agent)
    for _ in range(100):
        pool_colors, pool_faces = attacker.roll("front", "short")
        attack = world_state.attack
        attack.pool_faces = pool_faces
        attack.pool_colors = pool_colors
        # Count which items are matched to check if they are all encoded
        matched_dice = [0] * len(pool_faces)
        world_state.updateAttack(attack)
        # Make a random roll and encode the attack state
        # [ color - 3, face - 6]
        enc_attack = Encodings.encodeAttackState(world_state)[0][0]
        # Try to find a match for each color,face pair
        for slot in range(Encodings.max_die_slots):
            begin = dice_begin + slot * Encodings.hot_die_size
            end = begin + Encodings.hot_die_size
            dice_section = enc_attack[begin:end]
            # Should have a face and color or be empty
            assert (0 == dice_section.sum() or 2 == dice_section.sum())

            # There should be a color and a face
            if 0 != dice_section.sum():
                die_color = "none"
                for offset, color in enumerate(ArmadaDice.die_colors):
                    if 1.0 == dice_section[offset]:
                        die_color = color
                assert "none" != die_color

                die_face = "none"
                for offset, face in enumerate(ArmadaDice.die_faces):
                    if 1.0 == dice_section[len(ArmadaDice.die_colors) + offset]:
                        die_face = face
                assert "none" != die_face

                for idx, matched in enumerate(matched_dice):
                    if (0 == matched and pool_faces[idx] == die_face and
                        pool_colors[idx] == die_color):
                        matched_dice[idx] = 1
                        break
        # All dice should have been matched
        assert len(pool_faces) == sum(matched_dice)
