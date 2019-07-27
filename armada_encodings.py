#
# Copyright Bernhard Firner, 2019
#
# Encodings of the world state used with neural network models.
# 

from game_constants import (ArmadaPhases, ArmadaTypes)

class Encodings():

    # Make some constants here to avoid magic numbers in the rest of this code
    max_defense_tokens = ArmadaTypes.max_defense_tokens
    hot_token_size = len(ArmadaTypes.defense_tokens) + len(ArmadaTypes.token_colors) + 2
    hot_die_size = 9
    max_die_slots = 16

    def calculateAttackSize():
        attack_enc_size = 0
        # Encode the dice pool and faces, defense tokens, shields, and hull.
        # The input is much simpler if it is of fixed size. The inputs are:
        # attacker hull and shields - 5
        # defender hull and shields - 5
        attack_enc_size += 2*(len(ArmadaTypes.hull_zones) + 1)
        # 
        # Need to encode each tokens separately, with a one-hot vector for the token type, color,
        # and whether it was targetted with an accuracy. The maximum number of tokens is six, for
        # the SSD huge ship. The order of the tokens should be randomized with each encoding to
        # ensure that the model doesn't learn things positionally or ignore later entries that are
        # less common.
        # spent token types - 5
        attack_enc_size += len(ArmadaTypes.defense_tokens)
        # max_defense_tokens * [type - 5, color - 2, accuracy targeted - 1, spent - 1] = 48
        attack_enc_size += ArmadaTypes.max_defense_tokens * Encodings.hot_token_size
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

    def calculateSpendDefenseTokensSize():
        def_token_size = 0
        # Index of the token to spend with an output for no token
        def_token_size += ArmadaTypes.max_defense_tokens + 1
        # For evade target(s)
        def_token_size += Encodings.max_die_slots
        # For redirect target(s). The value in an index is the redirect amount
        def_token_size += len(ArmadaTypes.hull_zones)

        return def_token_size

