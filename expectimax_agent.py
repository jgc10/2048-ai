import math

class ExpectimaxAgent:
    def __init__(self, depth=3):
        self.depth = depth


    def get_best_move(self, game):
        legalMoves = game.get_legal_moves()

        if not legalMoves:
            return None

        bestMove = None
        bestValue = -float("inf")

        for move in legalMoves:
            successor = self.generateSuccessor(game, move) # simulate the move to generate the successor state
            value = self.expectimax(successor, 0, agentIndex=1) # evaluate state through expectimax.

            #keep highest value move

            if value > bestValue:
                bestValue = value
                bestMove = move

        return bestMove

    # creates new game state and applies move to it
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

    #recursive expectimax algorithm
    def expectimax(self, state, depth, agentIndex):

        if state.is_game_over() or depth == self.depth:
            return self.evaluate(state)

        if agentIndex == 0:
            return self.max_value(state, depth)
        else:
            return self.chance_value(state, depth)

    #chooses the best move
    def max_value(self, state, depth):
        moves = state.get_legal_moves()
        if not moves:
            return self.evaluate(state)

        values = []

        # calulates expectimax value
        for move in moves:
            successor = self.generateSuccessor(state, move)
            value = self.expectimax(successor, depth, agentIndex=1)
            values.append(value)

        return max(values)

    #Chance node: random spawning of tiles
    def chance_value(self, state, depth):
        empty = state.get_empty_tiles()
        if not empty:
            return self.evaluate(state)

        expected_value = 0

        # For each empty tile: tile can be 2 (p=0.9) or 4 (p=0.1)
        prob_each_tile = 1 / len(empty)

        for (x, y) in empty:
            #tile 2 = 90% chance
            s2 = state.copy()
            s2.board[x][y] = 2
            expected_value += 0.9 * prob_each_tile * self.expectimax(s2, depth + 1, 0)

            # tile 4 = 10%
            s4 = state.copy()
            s4.board[x][y] = 4
            expected_value += 0.1 * prob_each_tile * self.expectimax(s4, depth + 1, 0)

        return expected_value

    #evaluation: higher numbers = better score
    def evaluate(self, game):
        board = game.board

        empty_tiles = len(game.get_empty_tiles())
        max_tile = max(max(row) for row in board)

        return (
            empty_tiles * 200 +
            math.log(max_tile, 2) * 150
        )
