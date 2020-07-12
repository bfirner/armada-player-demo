# Should be run with pytest:
# > python3 -m pytest

import torch.multiprocessing as multiprocessing
import numpy
import pytest
import random
import torch

import ship
import utility
from armada_encodings import (Encodings)
from dice import (ArmadaDice)
from game_constants import (ArmadaTypes)
from learning_agent import (LearningAgent)
from random_action_dataset import (RandomActionDataset)
from random_agent import (RandomAgent)
from world_state import (AttackState, WorldState)

# Initialize ships from the test ship list
keys, ship_templates = utility.parseShips('data/test_ships.csv')

# Check the token section of the encoding. It occurs after the two hull and shield sections.
# Skip the spent section as well
token_types = len(ArmadaTypes.defense_tokens)

def get_defender_tokens(encoding, color):
    """
    Returns:
        torch.Tensor: The attack token section.
    """
    offset, size = ship.Ship.get_index("{}_defense_tokens".format(color))
    # The attacking ship section comes after the defending ship section
    #offset += ship.Ship.encodeSize()
    return encoding[offset:offset + size]

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
    encoding, die_slot_mapping = Encodings.encodeAttackState(world_state)
    Encodings.inPlaceUnmap(encoding, die_slot_mapping)

    return encoding, world_state


def test_get_training_examples():
    """Test that training sample generation is working."""
    # Seed the RNG and make sure this test is deterministic
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The regular Python RNG will be seeded in the dataloader workers
    # Use a dataloader with the RandomActionDataset
    phase = "attack - resolve attack effects"
    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(
            dataset=RandomActionDataset(phase, 100), batch_size=batch_size, shuffle=False,
            sampler=None, batch_sampler=None, num_workers=2, pin_memory=True,
            drop_last=False, multiprocessing_context=None)

    train_samples = 0
    for batch in train_dataloader:
        train_samples += batch[0].size(0)
    assert 100 == train_samples

    # TODO FIXME Verify that the evaluation set is deterministically generated with each pass
    # through the dataloader
    # Use a dataloader with the RandomActionDataset
    phase = "attack - spend defense tokens"
    eval_dataloader = torch.utils.data.DataLoader(
            dataset=RandomActionDataset(phase, 100, deterministic=True),
            batch_size=batch_size, shuffle=False,
            sampler=None, batch_sampler=None, num_workers=2, pin_memory=True,
            drop_last=False, timeout=0, multiprocessing_context=None)

    eval_samples = 0
    first_tensor = None
    for batch in eval_dataloader:
        if first_tensor is None:
            first_tensor = batch[0][0].clone()
        eval_samples += batch[0].size(0)
    assert 100 == eval_samples

    for batch in eval_dataloader:
        assert torch.equal(first_tensor, batch[0][0])
        break


def test_token_encodings():
    """Test that the encoding is correct for different defense tokens."""

    agent = LearningAgent()

    no_token = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=1)
    one_brace = ship.Ship(name="Single Brace", template=ship_templates["Single Brace"], upgrades=[], player_number=2)
    two_brace = ship.Ship(name="Double Brace", template=ship_templates["Double Brace"], upgrades=[], player_number=3)
    two_redirect = ship.Ship(name="Double Redirect", template=ship_templates["Double Redirect"], upgrades=[], player_number=4)
    two_evade = ship.Ship(name="Double Evade", template=ship_templates["Double Evade"], upgrades=[], player_number=5)
    two_contain = ship.Ship(name="Double Contain", template=ship_templates["Double Contain"], upgrades=[], player_number=6)
    two_scatter = ship.Ship(name="Double Scatter", template=ship_templates["Double Scatter"], upgrades=[], player_number=7)

    # Encode some attack states
    enc_one_brace = make_encoding(no_token, one_brace, "short", agent)[0]
    enc_two_brace = make_encoding(one_brace, two_brace, "short", agent)[0]
    enc_two_redirect = make_encoding(two_brace, two_redirect, "short", agent)[0]
    enc_two_evade = make_encoding(two_redirect, two_evade, "short", agent)[0]
    enc_two_contain = make_encoding(two_evade, two_contain, "short", agent)[0]
    enc_two_scatter = make_encoding(two_contain, two_scatter, "short", agent)[0]
    
    # Order of tokens in the encoding
    # token_types = ["evade", "brace", "scatter", "contain", "redirect"]
    # Check the green token section
    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("brace")] = 1.0
    assert torch.allclose(get_defender_tokens(enc_one_brace, "green"), ttensor)

    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("brace")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_brace, "green"), ttensor)

    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("redirect")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_redirect, "green"), ttensor)

    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("evade")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_evade, "green"), ttensor)

    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("contain")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_contain, "green"), ttensor)

    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("scatter")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_scatter, "green"), ttensor)


def test_red_token_encodings():
    """Test that the encoding is correct for red defense tokens."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    two_brace = ship.Ship(name="Double Brace", template=ship_templates["Double Brace"], upgrades=[], player_number=2)
    two_redirect = ship.Ship(name="Double Redirect", template=ship_templates["Double Redirect"], upgrades=[], player_number=2)
    two_evade = ship.Ship(name="Double Evade", template=ship_templates["Double Evade"], upgrades=[], player_number=2)
    two_contain = ship.Ship(name="Double Contain", template=ship_templates["Double Contain"], upgrades=[], player_number=2)
    two_scatter = ship.Ship(name="Double Scatter", template=ship_templates["Double Scatter"], upgrades=[], player_number=2)


    two_brace.spend_token('brace', ArmadaTypes.green)
    two_brace.spend_token('brace', ArmadaTypes.green)
    two_redirect.spend_token('redirect', ArmadaTypes.green)
    two_redirect.spend_token('redirect', ArmadaTypes.green)
    two_evade.spend_token('evade', ArmadaTypes.green)
    two_evade.spend_token('evade', ArmadaTypes.green)
    two_contain.spend_token('contain', ArmadaTypes.green)
    two_contain.spend_token('contain', ArmadaTypes.green)
    two_scatter.spend_token('scatter', ArmadaTypes.green)
    two_scatter.spend_token('scatter', ArmadaTypes.green)

    enc_two_brace = make_encoding(attacker, two_brace, "short", agent)[0]
    enc_two_redirect = make_encoding(attacker, two_redirect, "short", agent)[0]
    enc_two_evade = make_encoding(attacker, two_evade, "short", agent)[0]
    enc_two_contain = make_encoding(attacker, two_contain, "short", agent)[0]
    enc_two_scatter = make_encoding(attacker, two_scatter, "short", agent)[0]
    
    # Check the red token section
    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("brace")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_brace, "red"), ttensor)

    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("redirect")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_redirect, "red"), ttensor)

    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("evade")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_evade, "red"), ttensor)

    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("contain")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_contain, "red"), ttensor)

    ttensor = torch.zeros(len(ArmadaTypes.defense_tokens))
    ttensor[ArmadaTypes.defense_tokens.index("scatter")] = 2.0
    assert torch.allclose(get_defender_tokens(enc_two_scatter, "red"), ttensor)


def test_spent_encodings():
    """Test that the encoding is correct for different defense tokens."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    defender = ship.Ship(name="Defender", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)

    encoding, world_state = make_encoding(attacker, defender, "short", agent)

    # The defender and attacker come first, then the accuracied tokens, then the spent tokens
    spent_begin = 2 * ship.Ship.encodeSize() + 2 * len(ArmadaTypes.defense_tokens)
    spent_end = spent_begin + len(ArmadaTypes.defense_tokens)

    # Verify that no tokens are marked spent by default
    assert torch.sum(encoding[spent_begin:spent_end]) == 0.

    # Spend all of the tokens
    for tidx, ttype in enumerate(ArmadaTypes.defense_tokens):
        world_state.attack.defender_spend_token(ttype, 'green')

    encoding, die_slot_mapping = Encodings.encodeAttackState(world_state)
    Encodings.inPlaceUnmap(encoding, die_slot_mapping)
    assert torch.sum(encoding[spent_begin:spent_end]).item() == len(ArmadaTypes.defense_tokens)

    # Try spending the tokens at different indices
    for tidx, ttype in enumerate(ArmadaTypes.defense_tokens):
        # Re-encode and then set the token to spent.
        attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)
        defender = ship.Ship(name="Defender", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)
        encoding, world_state = make_encoding(attacker, defender, "short", agent)
        world_state.attack.defender_spend_token(ttype, 'green')
        encoding, die_slot_mapping = Encodings.encodeAttackState(world_state)
        encoding = Encodings.inPlaceUnmap(encoding, die_slot_mapping)
        assert torch.sum(encoding[spent_begin:spent_end]).item() == 1.0
        assert encoding[spent_begin:spent_end][tidx].item() == 1.0


def test_accuracy_encodings():
    """Test that the encoding is correct for dice targetted by an accuracy."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    three_brace = ship.Ship(name="Double Brace", template=ship_templates["Triple Brace"], upgrades=[], player_number=2)

    # Make a brace token red
    three_brace.spend_token('brace', ArmadaTypes.green)

    enc_three_brace, world_state = make_encoding(attacker, three_brace, "short", agent)

    # Define the offsets for convenience
    token_begin = Encodings.getAttackTokenOffset()
    token_end = token_begin + ArmadaTypes.max_defense_tokens

    # Verify that no tokens are targeted at first 
    assert 0.0 == enc_three_brace[token_begin:token_end].sum()

    # Now make a token red and target it
    three_brace.spend_token('brace', ArmadaTypes.green)
    green_acc_begin = Encodings.getAttackTokenOffset()
    green_acc_end = green_acc_begin + len(ArmadaTypes.defense_tokens)
    red_acc_begin = Encodings.getAttackTokenOffset() + len(ArmadaTypes.defense_tokens)
    red_acc_end = red_acc_begin + len(ArmadaTypes.defense_tokens)
    world_state.attack.accuracy_defender_token(ArmadaTypes.defense_tokens.index('brace'), ArmadaTypes.red)
    encoding, die_mapping = Encodings.encodeAttackState(world_state)

    enc_three_brace = Encodings.inPlaceUnmap(encoding, die_mapping)

    # Verify that only the red token has the accuracy flag set
    assert encoding[red_acc_begin + ArmadaTypes.defense_tokens.index('brace')].item() == 1.
    assert encoding[red_acc_begin:red_acc_end].sum().item() == 1.
    assert encoding[green_acc_begin:green_acc_end].sum().item() == 0.

    # Target both remaining green tokens
    world_state.attack.accuracy_defender_token(ArmadaTypes.defense_tokens.index('brace'), ArmadaTypes.green)
    world_state.attack.accuracy_defender_token(ArmadaTypes.defense_tokens.index('brace'), ArmadaTypes.green)
    encoding, die_mapping = Encodings.encodeAttackState(world_state)
    enc_three_brace = Encodings.inPlaceUnmap(encoding, die_mapping)

    # Verify that two green and one red brace have the accuracy flag
    assert encoding[red_acc_begin + ArmadaTypes.defense_tokens.index('brace')].item() == 1.
    assert encoding[red_acc_begin:red_acc_end].sum().item() == 1.
    assert encoding[green_acc_begin + ArmadaTypes.defense_tokens.index('brace')].item() == 2.
    assert encoding[green_acc_begin:green_acc_end].sum().item() == 2.

def test_range_encodings():
    """Test that the encoding is correct for ranges."""

    agent = LearningAgent()
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[],
                         player_number=1)
    no_token = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"],
                         upgrades=[], player_number=2)

    range_begin = Encodings.getAttackRangeOffset()
    for offset, attack_range in enumerate(ArmadaTypes.ranges):
        enc_attack = make_encoding(attacker, no_token, attack_range, agent)[0]
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
        enc_attack = Encodings.encodeAttackState(world_state)[0]
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
                for offset, face in enumerate(ArmadaDice.die_faces_frequencies):
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
