#
# Copyright Bernhard Firner, 2019-2020
#
# Encodings of the world state used with neural network models.
# 

import random
import torch

from dice import (ArmadaDice)
from game_constants import (ArmadaPhases, ArmadaTypes)
from ship import (Ship)

class Encodings():

    # Make some constants here to avoid magic numbers in the rest of this code
    max_defense_tokens = ArmadaTypes.max_defense_tokens
    hot_die_size = 9
    max_die_slots = 16

    def encodeShip(ship, encoding):
        """
        Encode a ship into the provided tensor.

        Arguments:
            ship (Ship)          : Ship to encode
            encoding (Tensor)    : Torch tensor to encode into
        Returns:
            (offset slots): Offset into the tensor after the last element written.
        """
        encoding[:Ship.encodeSize()] = ship.encoding
        offset = Ship.encodeSize()

        return offset

    def calculateAttackSize():
        attack_enc_size = 0

        # Two ships (the attacker and defender) will need to be encoded.
        attack_enc_size += 2 * Ship.encodeSize()

        # Accuracy targetted defense tokens (double for both red and green types of tokens)
        attack_enc_size += 2 * len(ArmadaTypes.defense_tokens)

        # Token types spent by the defender
        attack_enc_size += len(ArmadaTypes.defense_tokens)

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
        return 2 * Ship.encodeSize() + 3 * len(ArmadaTypes.defense_tokens)

    def getAttackDiceOffset():
        """Get the offset of the dice section of the attack state."""
        return Encodings.getAttackRangeOffset() + len(ArmadaTypes.ranges)

    def getAttackTokenOffset():
        """Get the offset the of defense token section of the attack state."""
        return 2 * Ship.encodeSize()

    def unmapSection(section, slice_size, slots):
        """Unmap a section of the encoding in place.

        This is used to unshuffle the data that was shuffled to remove positional bias in training.

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

    def inPlaceUnmap(encoding, die_slots):
        """Unmap the random slot arrangement done for training.

        This should be used for human consumption since the shuffling is only required to remove
        positional bias before creating training data for a network.

        Args:
            encoding (torch.Tensor): The encoding
            die_slots (List[int])  : Die slot mapping (source to encoding index)
        Returns:
            (torch.Tensor)         : Encoding of the world state with original orderings.
        """
        # Unshuffle the dice
        dice_begin = Encodings.getAttackDiceOffset()
        dice_end = dice_begin + Encodings.hot_die_size * Encodings.max_die_slots
        Encodings.unmapSection(section=encoding[dice_begin:dice_end], slice_size=Encodings.hot_die_size, slots=die_slots)

        return encoding

    def encodeAttackState(world_state, encoding=None):
        """
        Args:
            world_state (WorldState) : Current world state
            encoding (torch.Tensor)  : Optional memory for in-place encoding. A new tensor is created
                                       if encoding is not provided.
        Returns:
            (torch.Tensor)           : Encoding of the world state used as network input.
            (List[int])              : Die slot mapping (source to encoding index)
        """
        attack_enc_size = Encodings.calculateAttackSize()
        if encoding is None:
            encoding = torch.zeros(attack_enc_size)
        elif encoding.size(0) != attack_enc_size:
            raise RuntimeError("Tensor given to encodeAttackState is not the expected size.")

        # Now populate the tensor
        attack = world_state.attack
        defender = attack.defender
        attacker = attack.attacker

        cur_offset = Encodings.encodeShip(defender, encoding)
        cur_offset += Encodings.encodeShip(attacker, encoding[cur_offset:])

        # Encode whether an accuracy has been spent on the defender tokens
        encoding[cur_offset:cur_offset + 2 * len(ArmadaTypes.defense_tokens)] = torch.Tensor(
            attack.accuracy_tokens)
        cur_offset += 2 * len(ArmadaTypes.defense_tokens)
        # Encode whether a type of token has been spent by the defender
        encoding[cur_offset:cur_offset + len(ArmadaTypes.defense_tokens)] = torch.Tensor(
            attack.spent_types)
        cur_offset += len(ArmadaTypes.defense_tokens)

        # Attack range
        for offset, arange in enumerate(ArmadaTypes.ranges):
            encoding[cur_offset + offset] = 1 if arange == attack.range else 0
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
            color = attack.pool_colors[die_idx]
            encoding[slot_offset + ArmadaDice.die_colors.index(color)] = 1
            slot_offset += len(ArmadaDice.die_colors)
            # Encode die faces
            face = attack.pool_faces[die_idx]
            encoding[slot_offset + ArmadaDice.die_faces.index(face)] = 1

        # Sanity check on the encoding size and the data put into it
        assert encoding.size(0) == cur_offset + Encodings.hot_die_size * Encodings.max_die_slots

        return encoding, die_slots

    def calculateSpendDefenseTokensSize():
        def_token_size = 0
        # TODO FIXME Change to token types instead of token offsets
        # Index of the token to spend with an output for no token
        def_token_size += 2 * len(ArmadaTypes.defense_tokens)
        # For evade target(s)
        def_token_size += Encodings.max_die_slots
        # For redirect target(s). The value in an index is the redirect amount
        def_token_size += len(ArmadaTypes.hull_zones)

        return def_token_size

    def getSpendDefenseTokensEvadeOffset():
        """Get the offset to the evade target section."""
        return 2 * len(ArmadaTypes.defense_tokens)

    def getSpendDefenseTokensRedirectOffset():
        """Get the redirect zone targets offset."""
        return 2 * len(ArmadaTypes.defense_tokens)

    def calculateWorldStateSize():
        """Calculate the size of the world state encoding

        Returns:
            world state size (int)
        """

        # TODO FIXME This should probably accept an argument specifying the phase whose size should
        # be returned.
        # Only the current round is encoded at this moment.
        world_state_size = 1

        # TODO Encoding of the current main and sub phase

        # TODO Attack encoding (or 0s if not in an attack phase)

        # TODO Ship encodings

        # TODO This is the simplest possible thing to do while we are only concerned with attack
        # states, but must be fixed as we expand beyond that.
        return Encodings.calculateAttackSize() + world_state_size

    def encodeWorldState(world_state, encoding=None):
        """Calculate the encoding of the current world state.

        Arguments:
            world_state (WorldState): Current state.
            encoding (torch.Tensor) : Optional memory for in-place encoding. A new tensor is created
                                      if encoding is not provided.
        Returns:
            (torch.Tensor) : Encoding of the current state.
        """
        if encoding is None:
            encoding = torch.zeros(Encodings.calculateWorldStateSize())
        elif encoding.size(0) != Encodings.calculateWorldStateSize():
            raise RuntimeError("Tensor given to encodeWorldState is not the expected size.")

        # TODO Everything else needs to also be encoded obviously
        encoding[0] = world_state.round

        return encoding

    def calculateActionSize(subphase):
        """Calculate the encoding of an action in the given subphase.

        Arguments:
            subphase (str) : A string from the subphases in game_constants.py

        Returns:
            int : The encoding size. This is the size of a tensor to encode this action.
        """

        if "attack - resolve attack effects" == subphase:
            # Manipulate dice in the dice pool. Every ability or action that can occur needs to have
            # an encoding here.
            # TODO For now just handle spending accuracy icons. 
            # Each die could be an accuracy and could target any red or green token.
            return Encodings.max_die_slots + 2 * len(ArmadaTypes.defense_tokens)
            
        elif "attack - spend defense tokens" == subphase:
            # The defender spends tokens to modify damage or the dice pool. There are also other
            # abilities that can be used at this time, such as Admonition.
            # There are currently six token types. Some require additional targets.
            # No additional target required: "brace", "scatter", "contain", and "salvo"
            # The "evade" targets one (or more) of the attackers dice.
            # The "redirect" token targets one (or more with certain upgrades) hull zones.
            # TODO Just covering tokens for now.

            # The encoding begins with which types of tokens were spent (including red or green
            # state).
            token_size = 2 * len(ArmadaTypes.defense_tokens)
            # Evade indicates which dice were affected
            evade_size = Encodings.max_die_slots
            # Redirect indicates which hull zones were affected and the amount to redirect
            redirect_size = len(ArmadaTypes.hull_zones) + 1
            return token_size + evade_size + redirect_size
        else:
            raise NotImplementedError(
                "Encoding for attack phase {} not implemented.".format(subphase))

    def encodeAction(subphase, action_list, encoding=None):
        """Calculate the encoding of an action in the given subphase.

        Arguments:
            subphase (str)          : A string from the subphases in game_constants.py
            action_list ([str, ...]): A list of strings and arguments describing the actions.
            encoding (torch.Tensor) : Optional memory for in-place encoding. A new tensor is created
                                      if encoding is not provided.

        Returns:
            (torch.Tensor)      : Encoding of the action described by action_tuple.
        """
        if encoding is None:
            encoding = torch.zeros(Encodings.calculateActionSize(subphase))
        elif encoding.size(0) != Encodings.calculateActionSize(subphase):
            raise RuntimeError("Memory given to encodeAction is not the expected size.")

        if "attack - resolve attack effects" == subphase:
            # Manipulate dice in the dice pool. Every ability or action that can occur needs to have
            # an encoding here.
            # TODO For now just handle spending accuracy icons. 
            # Each die could be an accuracy and could target any of the defender's tokens.

            # Simple case when there was no action taken.
            if action_list is None or action_list[0] is None:
                return encoding
            action = action_list[0]
            if "accuracy" == action:
                token_begin = Encodings.max_die_slots
                for acc_action in action_list[1]:
                    die_index, token_index, color = acc_action
                    # Mark the token and spent die
                    encoding[die_index] = 1.
                    token_begin = token_index + color * len(ArmadaTypes.defense_tokens) + token_index
                    encoding[token_begin] += 1.
            else:
                raise NotImplementedError(
                    "Action {} unimplemented for attack phase {}.".format(action, subphase))
        elif "attack - spend defense tokens" == subphase:
            # The defender spends tokens to modify damage or the dice pool. There are also other
            # abilities that can be used at this time, such as Admonition.
            # There are currently six token types. Some require additional targets.
            # No additional target required: "brace", "scatter", "contain", and "salvo"
            # The "evade" targets one (or more) of the attackers dice.
            # The "redirect" token targets one (or more with certain upgrades) hull zones.
            # TODO Just covering tokens for now.

            # The action_list is actually a list of action tuples.
            if action_list is not None:
                for action_tuple in action_list:
                    # Offsets used during encodings
                    evade_offset = len(ArmadaTypes.defense_tokens) + ArmadaTypes.max_defense_tokens
                    redirect_offset = evade_offset + Encodings.max_die_slots

                    action = action_tuple[0]
                    action_args = action_tuple[1]
                    # Verify that we can handle this action
                    if action not in ArmadaTypes.defense_tokens and "none" != action:
                        raise NotImplementedError(
                            '"{}" unimplemented for attack phase {}.'.format(action, subphase))

                    # Handle the tokens that do not require targets
                    for ttype in ["brace", "scatter", "contain", "salvo"]:
                        if ttype == action:
                            token_index = action_args[0]
                            encoding[ArmadaTypes.defense_tokens.index(ttype)] = 1.
                            encoding[len(ArmadaTypes.defense_tokens) + token_index] = 1.
                    # Handle the tokens with targets
                    if "evade" == action:
                        token_index = action_args[0]
                        encoding[ArmadaTypes.defense_tokens.index(action)] = 1.
                        encoding[len(ArmadaTypes.defense_tokens) + token_index] = 1.
                        targets = action_args[1:]
                        for target in targets:
                            encoding[evade_offset + target] = 1.
                    elif "redirect" == action:
                        token_index = action_args[0]
                        encoding[ArmadaTypes.defense_tokens.index(action)] = 1.
                        encoding[len(ArmadaTypes.defense_tokens) + token_index] = 1.
                        targets = action_args[1]
                        for target, amount in targets:
                            encoding[redirect_offset + ArmadaTypes.hull_zones.index(target)] = amount
        else:
            raise NotImplementedError(
                "Encoding for attack phase {} not implemented.".format(subphase))
        return encoding


