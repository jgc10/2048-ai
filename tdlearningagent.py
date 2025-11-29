from game import Game
import statistics


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
        self.m = len(self.ntuples)
        self.learning_rate = 0.1
        self.LUT = {}

        # Initialize lookup tables
        for ntuple in self.ntuples:
            self.LUT[ntuple] = {}

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
    
    def v(self, ntuple: tuple[tuple[int]], state: Game) -> int:
        """
        The value function for a feature defined by an n-tuple. Feature weights are initialized
        to zero.

        :param ntuple: The coordinates of the tiles that define an n-tuple.
        :param state: The game state to read the tile values of.
        :return weight: The weight of the feature.
        """
        feature = tuple([state.board[x][y] for x, y in ntuple])

        if not feature in self.LUT[ntuple]:
            self.LUT[ntuple][feature] = 0
        
        return self.LUT[ntuple][feature]

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
                print(f"{move} value: {self.evaluate(state, move)}")
            
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
    
    def evaluate(self, state: Game, action: str) -> int:
        """
        Evaluate an action on a state by summing the reward and value of the afterstate. The value
        of the afterstate is the sum of all feature weights, as defined by the n-tuples, for all
        eight symmetries of the board state. E.g., if there are eight n-tuples, the value will be
        the sum of 8*8=64 features. 

        :param state: The game to take an action on.
        :param action: The action to take (LEFT, RIGHT, UP, DOWN).
        :return value: The reward + the value of the afterstate.
        """
        afterstate, reward = self.compute_afterstate(state, action)

        states = self.symmetries(afterstate)

        value = 0

        for s in states:
            for ntuple in self.ntuples:
                value += self.v(ntuple, s)

        return reward + value

    def learn_evaluation(self, state: Game, action: str, reward: int, afterstate: Game, next_state: Game):
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
        
        afterstates = self.symmetries(afterstate)

        for s in afterstates:
            for ntuple in self.ntuples:
                feature = tuple([s.board[x][y] for x, y in ntuple])
                self.LUT[ntuple][feature] = (
                    self.v(ntuple, s) + (self.learning_rate / self.m) * (next_reward + self.v(ntuple, next_afterstate) - self.v(ntuple, s))
                )
    
    def get_best_action(self, state: Game) -> str:
        """
        Returns the action that yields the most reward from the current state.

        :param state: The game to take an action on.
        :return action: The action with the greatest reward (LEFT, RIGHT, UP, DOWN). 
        """
        best = ("NULL", -999999)

        for action in state.get_legal_moves():
            value = self.evaluate(state, action)
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

        while not state.is_game_over():
            action = self.get_best_action(state)

            reward, afterstate, next_state = self.make_move(state, action)

            if not next_state.is_game_over():
                self.learn_evaluation(state, action, reward, afterstate, next_state)

            score += reward
            state = next_state

        return state


if __name__ == "__main__":
    agent = TdLearningAgent()

    scores = []
    tiles = []

    print("+--------------------------------------------------+")
    print("| Statistics from last 100 episodes:               |")
    print("|--------------------------------------------------|")
    print("| Episodes       | Mean Score     | Max Tile       |")
    print("|----------------|----------------|----------------|")

    for i in range(1, 10001):
        game = agent.play_game()
        scores.append(game.score)
        tiles.append(max(max(row) for row in game.board))

        # Print row every 100 episodes
        if i % 100 == 0:
            print("| {:>14} | {:>14.2f} | {:>14} |".format(i, statistics.mean(scores), max(tiles)))

            score = []
            tiles = []