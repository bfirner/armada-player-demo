__Changes__
* Added the beginnings of a learning agent
* * A model is actually trained to predict the turns until a ship is destroyed
* * Currently this is just supervised learning
* * Outputs are present for the defense tokens, but loss is not propagated so actions are arbitrary

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
* Switch from supervised learning to reinforcement learning
* * The model should predict an action and that should guide how tokens are spent
* * The model should simultaneously learn to predict the number of turns that the ship survives
* * Currently the poisson distribution is a better fit than the normal (it converges to a lower
    loss) but for other predictions this will not be the case.
* * Some network tuning could be done now to make the prediction better, and we should make a subset
    of the ships be for training and another set be for evaluation
* * The model can't predict that a ship will be destroyed when the roll will definately destroy it
* Train a model to use the defense tokens
* * Need to actually propagate loss for the defense token spending.
* Upgrades
* Location, movement, etc
* Command dials
* Squadrons
* Objectives

The current system can simulate the roll of dice but defense token usage is currently just a proof of concept. The next big goal is to use reinforcement learning to train a model to spend defense tokens. After that, progress will consist of programming the game state transitions and logic and then training models that play a more and more complete version of the game.
