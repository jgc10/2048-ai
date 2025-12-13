# 2048 AI

This repository is a collection of AI agent designed to play the game 2048. There are three agents:

1. TD learning agent
2. TD learning agent (experimental)
3. Expectimax agent

# TD Learning Agent

`tdlearning.py` implements the TD(0) reinforcement learning algorithm.

The agent determines the best move by evaluating afterstates using n-tuple networks. The afterstate is the state of a game board after the agent makes a move, but before a new tile spawns. N-tuple networks are required as the state space of 2048 is too large to evaluate every individual state. The 8x6 tuple network used by this agent is the optimal 8x6 network proposed by Matsuzaki.

The state evalation function evaluates all eight symmetries (rotational and reflective) of the board, and returns the sum of their values. The score obtained by entering a state is used to update the values of that state's features. Episodes are learned backwards. By default, the agent uses a static learning rate of 0.1.

The agent's configuration and training data will automatically save to a file after a set number of episodes.

## Usage

The main function in `tdlearning.py` has three constants that determine the agent's training/playing:

`SAVE_FILE_NAME` (default null): an agent save file that will be used to either continue training or play the game.
`EPISODES_PER_SAVE` (default 5000): the interval of games played during training when the training data will be saved.
`LEARN` (default true): if true, the agent will update its policy after every game played; otherwise, it will play using its current policy.

To run the agent:

```
$ python3 tdlearningagent.py
```

## References

M. Szubert and W. Jaśkowski, "Temporal difference learning of N-tuple networks for the game 2048," 2014 IEEE Conference on Computational Intelligence and Games, Dortmund, Germany, 2014, pp. 1-8, doi: 10.1109/CIG.2014.6932907.

K. Matsuzaki, "Systematic selection of N-tuple networks with consideration of interinfluence for game 2048," 2016 Conference on Technologies and Applications of Artificial Intelligence (TAAI), Hsinchu, Taiwan, 2016, pp. 186-193, doi: 10.1109/TAAI.2016.7880154.

W. Jaśkowski, "Mastering 2048 With Delayed Temporal Coherence Learning, Multistage Weight Promotion, Redundant Encoding, and Carousel Shaping," in IEEE Transactions on Games, vol. 10, no. 1, pp. 3-14, March 2018, doi: 10.1109/TCIAIG.2017.2651887.

H. Guei, L. -P. Chen and I. -C. Wu, "Optimistic Temporal Difference Learning for 2048," in IEEE Transactions on Games, vol. 14, no. 3, pp. 478-487, Sept. 2022, doi: 10.1109/TG.2021.3109887.

# Experimental TD Learning

TODO

# Expectimax Agent

TODO
