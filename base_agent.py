#! /usr/bin/python3

#
# Copyright Bernhard Firner, 2019
#
# The agent must choose which actions to take. An agent takes the form of:
#     (world state, current step, active player) -> action
#
# There are quite a few steps in Armada.

class BaseAgent:
    """The base agent. Replace the internal functions to implement your own agent."""

    def __init__(self, handlers):
        """Initialize state handler table.
        
        Arguments:
            handlers (Table(str, function)): A table that maps from phase names to a handler
                                             function.
        """
        self.handlers = handlers

    def default_handler(self, worldl_state):
        """The default handler called when a phase is not matched in the handlers table."""
        raise RuntimeError("Unhandled phase: {}", world_state["phase"])

    def handle(self, world_state):
        """Handles the given world state by passing it off to an internal function.
        
        Arguments:
            world_state (table): The collection of game state information.
        Returns:
            action to take
        """

        if world_state.full_phase in self.handlers.keys():
            return self.handlers[world_state.full_phase](world_state)
        else:
            return default_handler(world_state)

