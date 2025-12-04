from game import Game
import statistics
import time
import pickle


class TdLearningAgent:
    """
    Temporal difference learning agent.
    """
    def __init__(self):
        self.ntuples = (
            ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
            ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)),
            ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)),
            ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)),
            ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)),
            ((0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)),
            ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)),
            ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2))
        )
        """self.ntuples = (
            (1, 5, 9, 10, 13, 14),
            (2, 6, 10, 11, 14, 15),
            (2, 3, 6, 7, 10, 11),
            (3, 4, 7, 8, 11, 12)
        )"""
        self.m = len(self.ntuples)
        self.LUT = {}

        self.learn = True           # True => update LUTS after episode
        self.learning_rate = 0.1

        # Convert indices to coordinates if needed
        if type( self.ntuples[0][0] ) is int:
            new_ntuples = []
            for ntuple in self.ntuples:
                new_ntuples.append( self.make_ntuple(ntuple) )
            self.ntuples = tuple(new_ntuples)

        # Initialize lookup tables
        for ntuple in self.ntuples:
            self.LUT[ntuple] = {}
        
    def make_ntuple(self, indices: tuple[int]) -> tuple[tuple[int]]:
        """
        Convert tile indices to coordinates. Makes creating new n-tuples more convenient.

        +-------------------+
        | 1  | 2  | 3  | 4  |
        | 5  | 6  | 7  | 8  |
        | 9  | 10 | 11 | 12 |
        | 13 | 14 | 15 | 16 |
        +-------------------+

        :param indices: Tuple representing an n-tuple network.
        :return: An n-tuple network that can be used for the game. 
        """
        coordinates = []
        for i in indices:
            x = (i - 1) // 4
            y = (i - 1) % 4
            coordinates.append((x, y))
        
        return tuple(coordinates)

    def rotate(self, state: Game, n: int = 1) -> Game:
        """
        Rotate a game board 90 degrees clockwise.

        :param state: The game with the board to rotate.
        :param n: The number of rotations to perform.
        :return Game: The game with the rotated board.
        """
        rotated = state.copy()
        for _ in range(n):
            rotated.board = [list(reversed(col)) for col in zip(*rotated.board)]
        return rotated

    def mirror(self, state: Game):
        """
        Flip a game board over the horizontal axis.

        :param state: The game with the board to flip.
        :return Game: The game with the flipped board.
        """
        mirrored = state.copy()
        mirrored.board = mirrored.board[::-1]
        return mirrored
    
    def symmetries(self, state: Game) -> list[Game]:
        """
        Extract all eight symmetries (rotated and reflected) of a game state.

        :param state: The game with the board to obtain symmetries of.
        :return list[Game]: List of states that are all symmetries of each other.
        """
        states = [state]
        states.append( self.mirror(state) )

        for i in range(1, 4):
            rotated = self.rotate(state, i)
            states.append(rotated)
            states.append(self.mirror(rotated))

        return states

    def compute_afterstate(self, state: Game, action: str) -> tuple[Game, int]:
        """
        The afterstate is the state of the game after a move has been made, but
        before a new tile has been added to the board.

        :param state: The game's current state.
        :param action: The move to make on the board (LEFT, RIGHT, UP, DOWN).
        :return afterstate: The state of the game after the move, before a new tile is spawned.
        :return reward: The score gained from the move.
        """
        afterstate = state.copy()

        if action == "LEFT":
            reward = afterstate.move_left()
        elif action == "RIGHT":
            reward = afterstate.move_right()
        elif action == "UP":
            reward = afterstate.move_up()
        elif action == "DOWN":
            reward = afterstate.move_down()
        else:
            print(f"DEBUG: Invalid action.\nGame over: {state.is_game_over()}\nLegal moves: {state.get_legal_moves()}")

            for move in state.get_legal_moves():
                print(f"{move} value: {self.evaluate_action(state, move)}")
            
            state.print()
            raise ValueError(f"Invalid move: {action}")
        
        return afterstate, reward

    def make_move(self, state: Game, action: str) -> tuple[int, Game, Game]:
        """
        Simulates taking an action on a game, including spawning a new tile.

        :param state: The game to take an action on.
        :param action: The action to take (LEFT, RIGHT, UP, DOWN).
        :return reward: The score gained from the move.
        :return afterstate: The state of the game after the move, before a new tile is spawned.
        :return next state: The state of the game after the move, after a new tile is spawned.
        """
        afterstate, reward = self.compute_afterstate(state, action)
        
        next_state = afterstate.copy()
        next_state.spawn_tile()

        return reward, afterstate, next_state
    
    def evaluate_feature(self, ntuple: tuple[tuple[int]], state: Game) -> int:
        """
        Evaluate a feature of a state, which is defined by an n-tuple. Feature weights are
        initialized to zero.

        :param ntuple: The coordinates of the tiles that define an n-tuple.
        :param state: The game state to read the tile values of.
        :return weight: The weight of the feature.
        """
        feature = tuple([state.board[x][y] for x, y in ntuple])

        if not feature in self.LUT[ntuple]:
            self.LUT[ntuple][feature] = 0
        
        return self.LUT[ntuple][feature]

    def evaluate_state(self, state: Game) -> int:
        """
        Evaluate a state. The value of the afterstate is the sum of all feature weights, as
        defined by the n-tuples, for all eight symmetries of the board state. E.g., if there
        are eight n-tuples, the value will be the sum of 8*8=64 features.

        :param state: The game state to evaluate.
        :return value: The total value of the state.
        """
        states = self.symmetries(state)

        value = 0

        for s in states:
            for ntuple in self.ntuples:
                value += self.evaluate_feature(ntuple, s)

        return value
    
    def evaluate_action(self, state: Game, action: str) -> int:
        """
        Evaluate an action on a state by summing the reward and value of the afterstate.

        :param state: The game to take an action on.
        :param action: The action to take (LEFT, RIGHT, UP, DOWN).
        :return value: The reward + the value of the afterstate.
        """
        afterstate, reward = self.compute_afterstate(state, action)

        return reward + self.evaluate_state(afterstate)

    def learn_evaluation(self, state: Game, action: str, reward: int, afterstate: Game, next_state: Game) -> None:
        """
        Update the value of an afterstate.

        :param state: Unused.
        :param action: Unused.
        :param reward: Unused.
        :param afterstate: The afterstate to update the LUT value for.
        :param next_state: The state after adding a new tile to the afterstate.
        """
        next_action = self.get_best_action(next_state)
        next_afterstate, next_reward = self.compute_afterstate(next_state, next_action)

        afterstate_value = self.evaluate_state(afterstate)
        next_afterstate_value = self.evaluate_state(next_afterstate)
        
        for ntuple in self.ntuples:
            feature = tuple([afterstate.board[x][y] for x, y in ntuple])
            self.LUT[ntuple][feature] = (
                self.evaluate_feature(ntuple, afterstate)
                + (self.learning_rate / self.m)
                * (next_reward + next_afterstate_value - afterstate_value)
            )
    
    def get_best_action(self, state: Game) -> str:
        """
        Returns the action that yields the most reward from the current state.

        :param state: The game to take an action on.
        :return action: The action with the greatest reward (LEFT, RIGHT, UP, DOWN). 
        """
        best = ("NULL", -999999)

        for action in state.get_legal_moves():
            value = self.evaluate_action(state, action)
            if value > best[1]:
                best = (action, value)
        
        return best[0]
    
    def play_game(self) -> Game:
        """
        The main training loop. Starts a new game, makes decisions by querying the LUTS
        and learns by updating their values.

        :return state: The state of the finished game.
        """
        state = Game()
        score = 0
        history = []

        while not state.is_game_over():
            action = self.get_best_action(state)

            reward, afterstate, next_state = self.make_move(state, action)
            history.append( (afterstate, next_state) )

            score += reward
            state = next_state
        
        if self.learn:
            for afterstate, next_state in list(reversed(history)):
                if not next_state.is_game_over():
                    self.learn_evaluation(state, action, reward, afterstate, next_state)

        return state
    
    def save_agent(self, filename: str = "td_agent.pkl") -> None:
        """
        Saves the current configuration and training data to a file.
        
        :param filename: Name of the file to save to.
        """
        with open(f"saves/{filename}", "wb") as file:
            pickle.dump(self.learning_rate, file)
            pickle.dump(self.ntuples, file)
            pickle.dump(self.LUT, file)
    
    def load_agent(self, filename: str = "td_agent.pkl") -> None:
        """
        Loads an agent's configuration and training data from a file.
        
        :param filename: Name of the file to load from.
        """
        with open(f"saves/{filename}", "rb") as file:
            self.learning_rate = pickle.load(file)
            self.ntuples = pickle.load(file)
            self.LUT = pickle.load(file)
            self.m = len(self.ntuples)


if __name__ == "__main__":
    SAVE_FILE_NAME = ""
    EPISODES_PER_SAVE = 5000
    LEARN = True

    agent = TdLearningAgent()

    if SAVE_FILE_NAME:
        try:
            agent.load_agent(SAVE_FILE_NAME)
            print(f"Agent data loaded from saves/{SAVE_FILE_NAME}")
        except:
            print(f"Failed to load agent from save: saves/{SAVE_FILE_NAME}")
            exit()

    scores = []
    tiles = []
    start_time = time.time()
    
    if LEARN:
        agent.learn = True

        print("+----------------------------------------------------------------------------+")
        print("| Statistics from last 100 episodes:                                         |")
        print("|----------------------------------------------------------------------------|")
        print("| Episodes   | Time       | Mean Score     | Mean Max Tile  | Max Tile       |")
        print("|------------|------------|----------------|----------------|----------------|")

        for i in range(1, 100001):
            game = agent.play_game()
            scores.append(game.score)
            tiles.append(max(max(row) for row in game.board))

            # Save agent
            if i % EPISODES_PER_SAVE == 0:
                agent.save_agent(f"td_agent_{len(agent.ntuples)}x{len(agent.ntuples[0])}_{i}.pkl")

            # Print row every 100 episodes
            if i % 100 == 0:
                end_time = time.time()

                print("| {:>10} | {:>10.2f} | {:>14.2f} | {:>14.2f} | {:>14} |".format(
                    i,
                    end_time - start_time,
                    statistics.mean(scores),
                    statistics.mean(tiles),
                    max(tiles)
                ))

                score = []
                tiles = []
                start_time = time.time()
    
    else:
        agent.learn = False

        print("+-----------------------------------------------------------------------------------------+")
        print("| Cumulative Gameplay Statistics:                                                         |")
        print("|-----------------------------------------------------------------------------------------|")
        print("| # of Games | Time     | Mean Score     | Rate of Highest Tile                           |")
        print("|------------|----------|----------------|------------------------------------------------|")

        for i in range(1, 10001):
            game = agent.play_game()
            scores.append(game.score)
            tiles.append(max(max(row) for row in game.board))

            # Print row every 100 episodes
            if i % 100 == 0:
                end_time = time.time()

                max_tile_1_rate = tiles.count(max(tiles)) / len(tiles)
                max_tile_2_rate = tiles.count(max(tiles) / 2) / len(tiles)
                max_tile_3_rate = tiles.count(max(tiles) / 4) / len(tiles)

                print("| {:>10} | {:>8.2f} | {:>14.2f} | {:>5}: {:>6.2f}%, {:>5}: {:>6.2f}%, {:>5}: {:>6.2f}% |".format(
                    i,
                    end_time - start_time,
                    statistics.mean(scores),
                    max(tiles),
                    max_tile_1_rate * 100,
                    max(tiles) // 2,
                    max_tile_2_rate * 100,
                    max(tiles) // 4,
                    max_tile_3_rate * 100
                ))