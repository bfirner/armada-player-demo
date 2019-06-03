__Changes__
* Added the beginnings of a learning agent
* * The messiest bit, encoding an attack state, is done.
* * All tests except for the die encodings are complete.
* * The tests have some magic numbers in common with the agent, these should be shared through the
    agent (for example the size of some of the one-hot encodings)
* Started moving a lot of constants into the `game_constants` file
* * Will take some discipline to get all constants moved over

__Current State__
* Can encode the attack state part of the world model for use with a neural network
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
* Need to finish writing tests for the encodings
* Have a test for attacking with defense tokens, should have more to cover new functionality.
* Train a model to use the defense tokens
* *  Probably using reinforcement learning since we do not know the optimal policy.
* Upgrades
* Location, movement, etc
* Command dials
* Squadrons
* Objectives

The current system can simulate the roll of dice but defense token usage is currently just a proof of concept. The next big goal is to use reinforcement learning to train a model to spend defense tokens. After that, progress will consist of programming the game state transitions and logic and then training models that play a more and more complete version of the game.
