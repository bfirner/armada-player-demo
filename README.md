__Changes__
* Created a base agent class and extended with the sipmle agent.
* Create a real world state.
* Restricted a lot of functionality to a surrounding class (e.g. only the game agent is modifying
  the world state). This gets itneractions closer to their intended final states.

__Current State__
* Can parse and load ship information, although currently only the dice, shields, and hull zone information is used.
* Random rolling works.
* A basic simulator can track damage to a ship over multiple rolls.
* There is a proof of concept defense token spender
* The defense token step is handled with a single call to `handleAttack` and the `resolve attack effects`
  and `spend defence tokens` steps is handled.
* * Iterative phases should go like this:
* * * Send the agent the current state
* * * Get back an action. If the action is a no action indicator move on.
* * * Otherwise modify the state according to the action and send the state back to the agent.
* * This is typical of many of the phases in armada where you need to see the result of a previous action before choosing the next
* * This will work nicely with a machine learning agent because it turns the problem into a classification output.

__TODOs__
* Have a test for attacking with defense tokens, should have more to cover new functionality.
* Train a model to use the defense tokens
* *  Probably using reinforcement learning since we do not know the optimal policy.
* Upgrades
* Location, movement, etc
* Command dials
* Squadrons
* Objectives

The current system can simulate the roll of dice but defense token usage is currently just a proof of concept.
