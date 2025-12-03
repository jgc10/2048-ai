import math

class ExpectimaxAgent:
    def __init__(self, depth=2):
        self.depth = depth
        self.cache = {}

    def get_best_move(self, game):
        self.cache.clear()
        legalMoves = list(game.get_legal_moves())

        if not legalMoves:
            return None

        # Order moves by heuristic preference
        move_scores = []
        for move in legalMoves:
            successor = self.generateSuccessor(game, move)
            # Quick evaluation -- Helps with speed of the game allowing for greater depth usage
            score = self.quick_eval(successor)
            move_scores.append((score, move))
        # Try best moves first
        move_scores.sort(reverse=True)

        bestMove = None
        bestValue = -float("inf")
        for score, move in move_scores:
            successor = self.generateSuccessor(game, move)
            value = self.expectimax(successor, 0, agentIndex=1)
            if value > bestValue:
                bestValue = value
                bestMove = move

        return bestMove

    def quick_eval(self, state):
        board = state.board
        empty = len(state.get_empty_tiles())
        max_tile = max(max(row) for row in board)
        # keep max tile in corner
        corner_score = 0
        if board[0][0] == max_tile or board[0][3] == max_tile or \
           board[3][0] == max_tile or board[3][3] == max_tile:
            corner_score = 1000

        return empty * 100 + corner_score + max_tile

    def expectimax(self, state, depth, agentIndex):
        if state.is_game_over() or depth == self.depth:
            return self.evaluate(state)

        board_tuple = self.board_to_tuple(state)
        cache_key = (board_tuple, depth, agentIndex)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if agentIndex == 0:
            result = self.max_value(state, depth)
        else:
            result = self.chance_value(state, depth)

        self.cache[cache_key] = result
        return result

    def board_to_tuple(self, state):
        """Convert board to hashable tuple"""
        return tuple(tuple(row) for row in state.board)

    def max_value(self, state, depth):
        moves = list(state.get_legal_moves())
        if not moves:
            return self.evaluate(state)

        maxValue = -float("inf")

        for move in moves:
            successor = self.generateSuccessor(state, move)
            value = self.expectimax(successor, depth, agentIndex=1)
            maxValue = max(maxValue, value)

        return maxValue

    def generateSuccessor(self, game, move):
        new_game = game.copy()

        if move == "LEFT":
            new_game.move_left()
        elif move == "RIGHT":
            new_game.move_right()
        elif move == "UP":
            new_game.move_up()
        elif move == "DOWN":
            new_game.move_down()

        return new_game

    def chance_value(self, state, depth):
        empty = list(state.get_empty_tiles())
        if not empty:
            return self.evaluate(state)
        if len(empty) > 4:
            # Prioritize corner/edge positions
            priority_empty = []
            for pos in empty:
                x, y = pos
                # Corners and edges
                if (x in [0, 3] and y in [0, 3]) or (x in [0, 3] or y in [0, 3]):
                    priority_empty.append(pos)
            
            if len(priority_empty) > 4:
                empty = priority_empty[:4]
            else:
                empty = (priority_empty + empty)[:4]

        expected_value = 0
        prob_each_tile = 1.0 / len(empty)

        for (x, y) in empty:
            # Only for tile of 2 value
            s2 = state.copy()
            s2.board[x][y] = 2
            val2 = self.expectimax(s2, depth + 1, 0)
            expected_value += 0.9 * prob_each_tile * val2
            
            # Tile 4 if not too deep
            if depth < self.depth - 1:
                s4 = state.copy()
                s4.board[x][y] = 4
                val4 = self.expectimax(s4, depth + 1, 0)
                expected_value += 0.1 * prob_each_tile * val4
            else:
                # Approximation: tile 4 is 2x tile 2
                expected_value += 0.1 * prob_each_tile * val2 * 1.2

        return expected_value

    def evaluate(self, game):
        board = game.board
        empty_tiles = len(game.get_empty_tiles())
        # Find max tile and its position
        max_tile = 0
        max_pos = (0, 0)
        for i in range(4):
            for j in range(4):
                if board[i][j] > max_tile:
                    max_tile = board[i][j]
                    max_pos = (i, j)
        score = 0
        # empty tile bonus
        score += empty_tiles * 500 
        # max tile bonus
        score += max_tile * 2
        # corner bonus
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        if max_pos in corners:
            score += max_tile * 3
        else:
            score -= max_tile * 2
        mono_score = 0
        # monotonicity - if the max tile is in a corner
        '''Can be commented out for better speed
            But also helps the  game play better
            Tradeoff that has to be dealt with for now'''
        corner_dirs = {
            (0, 0): (1, 1),
            (0, 3): (-1, 1),
            (3, 0): (1, -1),
            (3, 3): (-1, -1)
        } # speed mapping
        if max_pos in corner_dirs:
            dx, dy = corner_dirs[max_pos]
            r = max_pos[0]
            c = max_pos[1]
            # Row monotonicity
            for j in range(3):
                if board[r][j] * dx >= board[r][j + 1] * dx:
                    mono_score += 100
            # Column monotonicity
            for i in range(3):
                if board[i][c] * dy >= board[i + 1][c] * dy:
                    mono_score += 100
        # adjacent tile bonus
        merge_bonus = 0
        for i in range(4):
            for j in range(4):
                if board[i][j] > 0:
                    # right
                    if j < 3 and board[i][j] == board[i][j+1]:
                        merge_bonus += board[i][j]
                    # down
                    if i < 3 and board[i][j] == board[i+1][j]:
                        merge_bonus += board[i][j]
        score += merge_bonus
        return score