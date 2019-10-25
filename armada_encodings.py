#
# Copyright Bernhard Firner, 2019
#
# Encodings of the world state used with neural network models.
# 

import random
import torch

from dice import (ArmadaDice)
from game_constants import (ArmadaPhases, ArmadaTypes)

class Encodings():

    # Make some constants here to avoid magic numbers in the rest of this code
    max_defense_tokens = ArmadaTypes.max_defense_tokens
    # Each token has a type, a color, and whether it has been spent in the current spend defense
    # tokens step.
    hot_token_size = len(ArmadaTypes.defense_tokens) + len(ArmadaTypes.token_colors) + 1
    hot_die_size = 9
    max_die_slots = 16

    def calculateShipSize():
        """
        Calculate the number of elements required to encode a ship.
        """
        # hull and shields
        ship_enc_size = (len(ArmadaTypes.hull_zones) + 1)
        # 
        # Need to encode each tokens separately, with a one-hot vector for the token type, color,
        # and whether it was targetted with an accuracy. The maximum number of tokens is six, for
        # the SSD huge ship. The order of the tokens should be randomized with each encoding to
        # ensure that the model doesn't learn things positionally or ignore later entries that are
        # less common.
        # Leave space for each possible token.
        ship_enc_size += ArmadaTypes.max_defense_tokens * Encodings.hot_token_size
        return ship_enc_size

    def encodeShip(ship, attack, encoding):
        """
        Encode a ship into the provided tensor.

        Arguments:
            ship (Ship)          : Ship to encode
            attack (AttackState) : Attack state object
            encoding (Tensor)    : Torch tensor to encode into
        Returns:
            (offset, token slots): Offset into the tensor of the last element written, the
                                   mapping from defender defense tokens to slots.
        """
        # Current offset into the encoding
        cur_offset = 0

        # Hull zones will of course increase with huge ships
        # Future proof this code by having things basing math upon the length of this table
        hull_zones = ArmadaTypes.hull_zones

        # Shields and hull
        for offset, zone in enumerate(hull_zones):
            encoding[0, cur_offset + offset] = ship.shields[zone]
        encoding[0, cur_offset + len(hull_zones)] = ship.hull()
        cur_offset += len(hull_zones) + 1

        # Defense tokens

        # Hot encoding the defenders tokens
        # Each hot encoding is: [type - 5, color - 2, spent - 1]
        token_slots = random.sample(range(ArmadaTypes.max_defense_tokens), len(ship.defense_tokens))
        for token_idx, slot in enumerate(token_slots):
            slot_offset = cur_offset + slot * Encodings.hot_token_size
            # Process if the ship has a token for this slot
            if token_idx < len(ship.defense_tokens):
                token = ship.defense_tokens[token_idx]
                # Encode the token type
                for offset, ttype in enumerate(ArmadaTypes.defense_tokens):
                    encoding[0, slot_offset + offset] = 1 if ttype in token else 0
                slot_offset = slot_offset + len(ArmadaTypes.defense_tokens)
                # Encode the token color
                for offset, color in enumerate(ArmadaTypes.token_colors):
                    encoding[0, slot_offset + offset] = 1 if color == token[0:len(color)] else 0
                slot_offset = slot_offset + len(ArmadaTypes.token_colors)
                # Encode if this particular token has been spent
                encoding[0, slot_offset] = 1 if ship.spent_tokens[token_idx] else 0

        # Move the current encoding offset to the position after the token section
        cur_offset += Encodings.hot_token_size * ArmadaTypes.max_defense_tokens

        return cur_offset, token_slots

    def calculateAttackSize():
        attack_enc_size = 0

        # Two ships (the attacker and defender) will need to be encoded.
        attack_enc_size += 2 * Encodings.calculateShipSize()

        # Accuracy targetted defense tokens
        attack_enc_size += ArmadaTypes.max_defense_tokens

        # Encode the dice pool and faces, defense tokens, shields, and hull.
        # The input is much simpler if it is of fixed size. The inputs are:
        #
        # attack range - 3 (1-hot encoding)
        attack_enc_size += len(ArmadaTypes.ranges)
        #
        # We have a couple of options for the dice pool.
        # The simplest approach is to over-provision a vector with say 16 dice. There would be 3
        # color vectors and 6 face vectors for a total of 9*16=144 inputs.
        # 16 * [ color - 3, face - 6]
        attack_enc_size += Encodings.max_die_slots * Encodings.hot_die_size

        return attack_enc_size

    def getAttackRangeOffset():
        """Get the offset of the dice section of the attack state."""
        return 2 * Encodings.calculateShipSize() + ArmadaTypes.max_defense_tokens

    def getAttackDiceOffset():
        """Get the offset of the dice section of the attack state."""
        return 2 * Encodings.calculateShipSize() + ArmadaTypes.max_defense_tokens + len(ArmadaTypes.ranges)

    def getAttackTokenOffset():
        """Get the offset the of defense token section of the attack state."""
        return 2 * Encodings.calculateShipSize()

    def unmapSection(section, slice_size, slots):
        """Unmap a section of the encoding in place.

        Args:
            section (torch.Tensor): A section of an encoding
            slice_size       (int): The number of elements in each slice of the section.
            slots      (List[int]): Slot mapping (source to encoding index)
        """
        # Make a buffer to store the unshuffled section and copy each entry
        new_section = torch.zeros(section.size())
        for src_idx, encoding_idx in enumerate(slots):
            slot_begin = encoding_idx * slice_size
            slot_end = slot_begin + slice_size

            src_begin = src_idx * slice_size
            src_end = src_begin + slice_size

            new_section[src_begin:src_end] = section[slot_begin:slot_end]
        # Assign back to the given section
        section[0:] = new_section

    def inPlaceUnmap(encoding, token_slots, die_slots):
        """Unmap the random slot arrangement done for training.

        Args:
            encoding (torch.Tensor): The encoding
            token_slots (List[int]): Token slot mapping (source to encoding index)
            die_slots (List[int])  : Die slot mapping (source to encoding index)
        Returns:
            (torch.Tensor)         : Encoding of the world state with original orderings.
        """
        # Unshuffle the tokens
        token_begin = Encodings.getAttackTokenOffset()
        token_end = token_begin + ArmadaTypes.max_defense_tokens
        Encodings.unmapSection(encoding[0, token_begin:token_end], 1, token_slots)

        # Unshuffle the ship tokens
        def_tokens_begin = len(ArmadaTypes.hull_zones) + 1
        def_tokens_end = def_tokens_begin + ArmadaTypes.max_defense_tokens * Encodings.hot_token_size
        Encodings.unmapSection(encoding[0, def_tokens_begin:def_tokens_end], Encodings.hot_token_size, token_slots)

        # Unshuffle the dice
        dice_begin = Encodings.getAttackDiceOffset()
        dice_end = dice_begin + Encodings.hot_die_size * Encodings.max_die_slots
        Encodings.unmapSection(section=encoding[0, dice_begin:dice_end], slice_size=Encodings.hot_die_size, slots=die_slots)

        return encoding

    def encodeAttackState(world_state):
        """
        Args:
            world_state (WorldState) : Current world state
        Returns:
            (torch.Tensor)           : Encoding of the world state used as network input.
            (List[int])              : Token slot mapping (source to encoding index)
            (List[int])              : Die slot mapping (source to encoding index)
        """
        attack_enc_size = Encodings.calculateAttackSize()
        encoding = torch.zeros(1, attack_enc_size)

        # Now populate the tensor
        attack = world_state.attack
        defender = attack.defender
        attacker = attack.attacker

        cur_offset, token_slots = Encodings.encodeShip(defender, attack, encoding)
        cur_offset += Encodings.encodeShip(attacker, attack, encoding[:,cur_offset:])[0]

        # Encode whether an accuracy has been spent on the defender tokens
        for token_idx, slot in enumerate(token_slots):
            slot_offset = cur_offset + slot
            if token_idx < len(defender.defense_tokens):
                encoding[0, slot_offset] = 1 if attack.accuracy_tokens[token_idx] else 0
        cur_offset += ArmadaTypes.max_defense_tokens

        # Attack range
        for offset, arange in enumerate(ArmadaTypes.ranges):
            encoding[0, cur_offset + offset] = 1 if arange == attack.range else 0
        cur_offset += len(ArmadaTypes.ranges)

        # Each die will be represented with a hot_die_size vector
        # There are max_die_slots slots for these, and we will fill them in randomly
        # During training we want the model to react properly to a die in any location in the
        # vector, so we randomize the dice locations so that the entire vector is used (otherwise
        # the model would be poorly trained for rolls with a very large number of dice)
        die_slots = random.sample(range(Encodings.max_die_slots), len(attack.pool_colors))
        for die_idx, slot in enumerate(die_slots):
            slot_offset = cur_offset + slot * Encodings.hot_die_size
            # Encode die colors
            for offset, color in enumerate(ArmadaDice.die_colors):
                encoding[0, slot_offset + offset] = 1 if color == attack.pool_colors[die_idx] else 0
            slot_offset += len(ArmadaDice.die_colors)
            # Encode die faces
            for offset, face in enumerate([face for face in ArmadaDice.die_faces.keys()]):
                encoding[0, slot_offset + offset] = 1 if face == attack.pool_faces[die_idx] else 0

        # Sanity check on the encoding size and the data put into it
        assert encoding.size(1) == cur_offset + Encodings.hot_die_size * Encodings.max_die_slots

        # TODO FIXME HERE Ship encodings and the location or accuracied and spent tokens have
        # changed so the tests need to be updated
        return encoding, token_slots, die_slots

    def calculateSpendDefenseTokensSize():
        def_token_size = 0
        # Index of the token to spend with an output for no token
        def_token_size += ArmadaTypes.max_defense_tokens + 1
        # For evade target(s)
        def_token_size += Encodings.max_die_slots
        # For redirect target(s). The value in an index is the redirect amount
        def_token_size += len(ArmadaTypes.hull_zones)

        return def_token_size

    def calculateWorldStateSize():
        """Calculate the size of the world state encoding

        Returns:
            world state size (int)
        """

        # Start from 0 and build ourselves up
        world_state_size = 0

        # Encoding of the current main and sub phase

        # Attack encoding (or 0s if not in an attack phase)

        # Ship encodings

        return Encodings.calculateAttackSize() + world_state_size

