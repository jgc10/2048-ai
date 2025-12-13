# 2048 AI

This repository is a collection of AI agent designed to play the game 2048. There are three agents:

1. TD learning agent
2. TD learning agent (experimental)
3. Expectimax agent

# TD Learning Agent

`tdlearning.py` implements the TD(0) reinforcement learning algorithm.

The agent determines the best move by evaluating afterstates using n-tuple networks. The afterstate is the state of a game board after the agent makes a move, but before a new tile spawns. N-tuple networks are required as the state space of 2048 is too large to evaluate every individual state. The 8x6 tuple network used by this agent is the optimal 8x6 network proposed by Matsuzaki.

The state evalation function evaluates all eight symmetries (rotational and reflective) of the board, and returns the sum of their values. The score obtained by entering a state is used to update the values of that state's features. Episodes are learned backwards.

The agent's configuration and training data will automatically save to a file after a set number of episodes.

## Results

Below are the results of playing 2,500 games on 40,000 episodes of training with a learning rate of 0.1. These results are expected to improve with more episodes of training.

- Average score: 28,619
- Reached 8192: 0.04%
- Reached 4096: 9.52%​
- Reached 2048: 60.64%​

## Usage

The main function in `tdlearning.py` has three constants that determine the agent's training/playing:

- `SAVE_FILE_NAME` (default null): an agent save file that will be used to either continue training or play the game.
- `EPISODES_PER_SAVE` (default 5000): the interval of games played during training when the training data will be saved.
- `LEARN` (default true): if true, the agent will update its policy after every game played; otherwise, it will play using its current policy.

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

`experimentaltdlearning.py` implements the TD(0) reinforcement learning algorithm.

The agent determines the best move by evaluating afterstates using n-tuple networks. The afterstate is the state of a game board after the agent makes a move, but before a new tile spawns. N-tuple networks are required as the state space of 2048 is too large to evaluate every individual state. An 8x6 tuple network is combined with a 10x8 tuple network in an attempt to increase feature evaluations to increase the scoring of the agent when performing the game.

The agent also determines the move by evaluating numerous game heuristics that guide the agent to perform moves based on optimal values. The agent implements epsilon-greedy exploration to balance exploitation of learned knowledge.

## Results

Below are the results of playing 14,000 games on 120,000 episodes of training with a learning rate of 0.005. These results are expected to improve with more episodes of training and an increase in the learning rate.
​
- Average score: 16,693​
- Reached 4096 tile: 0.34%​
- Reached 2048 tile: 21%​
- Reached 1024 tile: 74.99%

## Usage
The main function in `tdlearning.py` has three variables that determine the agent's training/playing:

The main function in `tdlearning.py` has three constants that determine the agent's training/playing:

- `SAVE_FILE_NAME` (default null): an agent save file that will be used to either continue training or play the game.
- `EPISODES_PER_SAVE` (default 5000): the interval of games played during training when the training data will be saved.
- `LEARN` (default true): if true, the agent will update its policy after every game played; otherwise, it will play using its current policy.

To run the agent:

```
$ python3 tdlearningagent.py
```

# Expectimax Agent

`expectimaxagent.py` implements the Expectimax algorithim.

The agent simulates all possible legal moves and then ranks them using a heuristic. The Expectimax search then chooses the move that leads to the maximum score while determining the different chance nodes. To increase performance, previously seen states are cached to limit the necessary computations. This Expectimax algorithm can run depths 2-4 with reasonable timing.

Rewards were given for empty tiles, higher-number tiles in corners, and merge potential of multiple tiles.

## Key Parts

- `get_best_move`- entry point
- `expectimax`- algorithm controller
- `max_value`- AI decision
- `chance_value`- randomness modeling
- `evaluate`- scoring states

## Results

### Depth 2

- Average score: 8321
- Reached 2048: 0.87%
- Reached 1024: 19.57%
- Reached 512: 76.53%

### Depth 3

- Average score: 12569
- Reached 2048: 4%
- Reached 1024: 54%
- Reached 512: 95%

### Depth 4

- Average score: 18543
- Reached 2048: 30%
- Reached 1024: 48%
- Reached 512: 20%

## Usage
To run the agent:

```
$ python3 expectimaxagent.py
```

## References

T. Zhong, "AI for Game 2048", GitHub repository, 2017. Available: https://github.com/cczhong11/2048-ai/blob/master/AI%20for%202048%20write%20up.pdf
