__Changes__
* Fixed a lot of bugs in the previous commit
* Added an agent that can select accuraces to lock down defense tokens
* The agent's role has been updated to match what it must be to plug in different AI modules.
* * The agent no longer modifies the world state
* * The agent is called multiple times as long as it has choices to make and returns an empty result to indicate no more optional actions will be taken.
* The game state has a few more pieces of information in it, but there isn't any state logging yet.

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
* We need tests because bugs are easy to write and there is enough code here that we can worry about
  regressions
* * Basically we should do some clean up to remove some TODOs but there is always a chance to break
    something when you work on it
* * The best way to prevent this is to have a few tests
* Upgrades
* Location, movement, etc
* Command dials
* Squadrons
* Objectives

The current system can simulate the roll of dice but defense token usage is currently just a proof of concept.
