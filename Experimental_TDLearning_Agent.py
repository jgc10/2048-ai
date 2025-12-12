from game import Game
import statistics
import pickle
import random
import time

class TdLearningAgent:
    """
    Temporal difference learning agent.
    """
    def __init__(self):
        self.ntuples_6 = (
            ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
            ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)),
            ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)),
            ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)),
            ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)),
            ((0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)),
            ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)),
            ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2))
        )

        # 8-tile n-tuples 
        self.ntuples_8 = (
            # 2x4 row patterns
            ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)),
            ((2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)),
            
            # 4x2 column patterns
            ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)),
            ((0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3)),
            
            # L-shaped corner patterns
            ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1)),
            ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (2, 1), (1, 1)),
            ((0, 3), (0, 2), (0, 1), (1, 3), (2, 3), (3, 3), (1, 2), (2, 2)),
            ((3, 3), (2, 3), (1, 3), (0, 3), (3, 2), (3, 1), (2, 2), (1, 2)),
            
            # Diagonal patterns
            ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2)),
            ((0, 3), (0, 2), (1, 3), (1, 2), (1, 1), (2, 2), (2, 1), (3, 1))
        )

        self.ntuples = self.ntuples_6 + self.ntuples_8

        self.m = len(self.ntuples)
        self.learning_rate = 0.005
        self.LUT = {}

        self.epsilon = 0.01
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.99995

        self.learn = True

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
        states.append(self.mirror(state))

        for i in range(1, 4):
            rotated = self.rotate(state, i)
            states.append(rotated)
            states.append(self.mirror(rotated))

        return states
    
    def evaluate_feature(self, ntuple: tuple[tuple[int]], state: Game) -> int:
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

        #for s in states:
         #   for ntuple in self.ntuples:
          #      value += self.evaluate_feature(ntuple, s)
        for s in states:
            board = s.board
            for ntuple in self.ntuples:
                feature = tuple(board[x][y] for x, y in ntuple)
                lut = self.LUT[ntuple]
                if feature not in lut:
                    lut[feature] = 0
                
                value += lut[feature]
        return value

    def evaluate_action(self, state: Game, action: str) -> int:
        """
        Evaluate an action on a state by summing the reward and value of the afterstate.

        :param state: The game to take an action on.
        :param action: The action to take (LEFT, RIGHT, UP, DOWN).
        :return value: The reward + the value of the afterstate.
        """
        afterstate, reward = self.compute_afterstate(state, action)

        max_tile = 0
        max_pos = (0, 0)
        board = afterstate.board
        
        for i in range(4):
            for j in range(4):
                if board[i][j] > max_tile:
                    max_tile = board[i][j]
                    max_pos = (i, j)
        
        bonus = 0
        penalty = 0
        monotonicity_bonus = 0

        if max_pos in ((0, 0), (0, 3), (3, 0), (3, 3)):  # Tuple faster than list
            if max_tile >= 256:
                bonus = max_tile >> 1  # Bit shift faster than * 0.5
        else:
            if max_tile >= 512:
                penalty = -(max_tile * 4 // 5)  # Integer ops faster
        
        if max_tile >= 1024:
            bonus += self._tile_bonus(max_tile)
        '''for row in afterstate.board:
            if all(row[i] >= row[i+1] for i in range(3) if row[i] > 0):
                monotonicity_bonus += 500
            elif all(row[i] <= row[i+1] for i in range(3) if row[i+1] > 0):
                monotonicity_bonus += 500
    
        for col_idx in range(4):
            col = [afterstate.board[row][col_idx] for row in range(4)]
            if all(col[i] >= col[i+1] for i in range(3) if col[i] > 0):
                monotonicity_bonus += 500
            elif all(col[i] <= col[i+1] for i in range(3) if col[i+1] > 0):
                monotonicity_bonus += 500'''
        
        #+ monotonicity_bonus

        return reward + bonus + penalty  + self.evaluate_state(afterstate)
    
    def _tile_bonus(self, max_tile):
        """Lookup table for common tile bonuses"""
        if max_tile == 1024:
            return 200
        elif max_tile == 2048:
            return 2800
        elif max_tile == 4096:
            return 15300 
        else:
            # Fallback calculation
            bonus = 0
            if max_tile >= 1024:
                bonus += ((max_tile / 1024) ** 2) * 200
            if max_tile >= 2048:
                bonus += ((max_tile / 2048) ** 2) * 2000
            if max_tile >= 4096:
                bonus += ((max_tile / 4096) ** 2.5) * 10000
            return bonus

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

        max_before = max(max(row) for row in state.board)
        max_after = max(max(row) for row in afterstate.board)
        if max_after > max_before:
            if max_after >= 1024:
                reward += max_after
            if max_after == 2048:
                reward += 2048
            elif max_after == 4096:
                reward += 8192
            elif max_after == 8192:
                reward += 32768

        next_state = afterstate.copy()
        next_state.spawn_tile()

        return reward, afterstate, next_state 

    def learn_evaluation(self, state: Game, action: str, reward: int, afterstate: Game, next_state: Game):
        """
        Update the value of an afterstate.

        :param state: Unused.
        :param action: Unused.
        :param reward: Unused.
        :param afterstate: The afterstate to update the LUT value for.
        :param next_state: The state after adding a new tile to the afterstate.
        """
        next_action = self.get_best_action(next_state, explore=False)
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
    
    def get_best_action(self, state: Game, explore: bool = True) -> str:
        """
        Returns the action that yields the most reward from the current state.

        :param state: The game to take an action on.
        :return action: The action with the greatest reward (LEFT, RIGHT, UP, DOWN). 
        """
        legal_moves = list(state.get_legal_moves())
        
        best = ("NULL", -999999)
        if explore and random.random() < self.epsilon:
            return random.choice(legal_moves)

        for action in state.get_legal_moves():
            value = self.evaluate_action(state, action)
            if value > best[1]:
                best = (action, value)
        
        return best[0]
    
    def save_model(self, filename='td_agent_lut.pkl'):
        """Save the lookup tables to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.LUT, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename='td_agent_lut.pkl'):
        """Load the lookup tables from a file."""
        with open(filename, 'rb') as f:
            self.LUT = pickle.load(f)
        print(f"Model loaded from {filename}")
    
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
            if self.learn:
                if not next_state.is_game_over():
                    self.learn_evaluation(state, action, reward, afterstate, next_state)
            
            score += reward
            state = next_state
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return state


if __name__ == "__main__":
    agent = TdLearningAgent()
    '''
    Comment out the next line if a pickle file does not exist.
    Change the i-range for better display of the episode number
    in the table as well.
    '''
    agent.load_model('td_agent_episode_ft67000.pkl')

    scores = []
    tiles = []
    start_time = time.time()
    learn = False

    if learn == True:
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

            # Print row every 100 episodes
            if i % 100 == 0:
                end_time = time.time()

                print("| {:>10} | {:>10.2f} | {:>14.2f} | {:>14.2f} | {:>14} |".format(
                    i, end_time - start_time, statistics.mean(scores), statistics.mean(tiles), max(tiles)
                ))
                score = []
                tiles = []
                start_time = time.time()
            if i % 1000 == 0:
                agent.save_model(f'td_agent_episode_ft{i}.pkl')
    else:
        agent.learn = False
        print("+-----------------------------------------------------------------------------------------+")
        print("| Cumulative Gameplay Statistics:                                                         |")
        print("|-----------------------------------------------------------------------------------------|")
        print("| # of Games | Time     | Mean Score     | Rate of Highest Tile                           |")
        print("|------------|----------|----------------|------------------------------------------------|")

        for i in range(1, 50001):
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
        