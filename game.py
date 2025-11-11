import random
import copy


class Game:
    """
    A game of 2048.

    An instance of this class includes the board and score, and all methods used to modify
    the state of the game.
    """

    def __init__(self, size: int = 4) -> None:
        """
        Initialize the score to 0, and create the game board.

        Add two starting tile to the board.
        """
        self.score = 0

        self.size = size
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]

        # Initialize the board with two tiles
        self.spawn_tile()
        self.spawn_tile()

    def print(self) -> None:
        """
        Print the board in a formatted grid.
        """
        board_width = self.size * 5 + 1

        # Score header
        print("+" + "-" * board_width + "+")
        print(f"| Score: {self.score:<{board_width - 8}}" + "|")
        print("+" + "-" * board_width + "+")

        for row in self.board:
            print("|", end="")

            for tile in row:
                if tile == 0:
                    print("{:>5}".format("."), end="")      # print "." for empty tiles
                else:
                    print("{:>5}".format(tile), end="")
            
            print(" |")
        
        print("+" + "-" * board_width + "+")
    
    def get_empty_tiles(self) -> list[tuple[int, int]]:
        """
        Get a list of empty tiles.

        :return Empty tiles: List of (x, y) positions of empty tiles.
        """
        empty_tiles = []

        for x in range( len(self.board) ):
            for y in range( len(self.board[x]) ):
                if self.board[x][y] == 0:
                    empty_tiles.append( (x, y) )
        
        return empty_tiles
    
    def spawn_tile(self) -> None:
        """
        Adds a 2 or 4 tile to board at a random position.
        """
        new_tile = 2 if random.random() < 0.9 else 4        # 90% chance for 2 tile, 10% for 4 tile

        empty_tiles = self.get_empty_tiles()

        if empty_tiles:
            x, y = random.choice(empty_tiles)
            self.board[x][y] = new_tile
        else:
            raise ValueError("Cannot add new tile: board is full.")
    
    def get_legal_moves(self) -> set[str]:
        """
        Get the moves that can be made on the current board. Possible moves include: "LEFT", "RIGHT", "UP", "DOWN".

        :return Legal moves: Set of legal moves.
        """
        legal_moves = set()

        for row in self.board:
            for x in range(len(row) - 1):
                if (
                    (row[x] == 0 and row[x + 1] != 0) or
                    (row[x] != 0 and row[x] == row[x + 1])
                ):
                    legal_moves.add("LEFT")
                    break
            
            if "LEFT" in legal_moves:
                break
            
        for row in self.board:
            for x in range(len(row) - 1, 0, -1):
                if (
                    (row[x] == 0 and row[x - 1] != 0) or
                    (row[x] != 0 and row[x] == row[x - 1])
                ):
                    legal_moves.add("RIGHT")
                    break
            
            if "RIGHT" in legal_moves:
                break
        
        for i in range(self.size):
            col = [row[i] for row in self.board]
            for y in range(len(col) - 1):
                if (
                    (col[y] == 0 and col[y + 1] != 0) or
                    (col[y] != 0 and col[y] == col[y + 1])
                ):
                    legal_moves.add("UP")
                    break
            
            if "UP" in legal_moves:
                break
        
        for i in range(self.size):
            col = [row[i] for row in self.board]
            for y in range(len(col) - 1, 0, -1):
                if (
                    (col[y] == 0 and col[y - 1] != 0) or
                    (col[y] != 0 and col[y] == col[y - 1])
                ):
                    legal_moves.add("DOWN")
                    break
            
            if "DOWN" in legal_moves:
                break
        
        return legal_moves

    def is_game_over(self) -> bool:
        """
        Check if there are no more available moves.

        :return Game over: True the game is over, false otherwise.
        """

        if self.get_legal_moves():
            return False
        else:
            return True
    
    def move_left(self) -> int:
        """
        Moves tiles to the left and merges like tiles.

        :return Points: the # of points added to the total score from making this move.
        """
        new_score = 0

        for x, row in enumerate(self.board):
            row_values = [n for n in row if n != 0]
            tile_index = 0

            new_row = [0 for _ in range(self.size)]
            new_index = 0

            while tile_index < len(row_values):
                if (
                    tile_index < ( len(row_values) - 1) and
                    row_values[tile_index] == row_values[tile_index + 1]
                ):
                    new_row[new_index] = row_values[tile_index] * 2
                    new_score += new_row[new_index]
                    new_index += 1
                    tile_index += 2
                else:
                    new_row[new_index] = row_values[tile_index]
                    new_index += 1
                    tile_index += 1
            
            self.board[x] = new_row

        self.score += new_score
        return new_score
    
    def move_right(self) -> int:
        """
        Moves tiles to the right and merges like tiles.

        :return Points: the # of points added to the total score from making this move.
        """
        new_score = 0

        for x, row in enumerate(self.board):
            row_values = [n for n in row if n != 0]
            tile_index = len(row_values) - 1
            
            new_row = [0 for _ in range(self.size)]
            new_index = self.size - 1

            while tile_index >= 0:
                if (
                    tile_index > 0 and
                    row_values[tile_index] == row_values[tile_index - 1]
                ):
                    new_row[new_index] = row_values[tile_index] * 2
                    new_score += new_row[new_index]
                    new_index -= 1
                    tile_index -= 2
                else:
                    new_row[new_index] = row_values[tile_index]
                    new_index -= 1
                    tile_index -= 1
            
            self.board[x] = new_row
        
        self.score += new_score
        return new_score
    
    def move_up(self) -> int:
        """
        Moves tiles up and merges like tiles.

        :return Points: the # of points added to the total score from making this move.
        """
        new_score = 0

        for y in range(self.size):
            col_values = [
                row[y] for row in self.board if row[y] != 0
            ]
            tile_index = 0

            new_col = [0 for _ in range(self.size)]
            new_index = 0

            while tile_index < len(col_values):
                if (
                    tile_index < ( len(col_values) - 1) and
                    col_values[tile_index] == col_values[tile_index + 1]
                ):
                    new_col[new_index] = col_values[tile_index] * 2
                    new_score += new_col[new_index]
                    new_index += 1
                    tile_index += 2
                else:
                    new_col[new_index] = col_values[tile_index]
                    new_index += 1
                    tile_index += 1
            
            for j in range(self.size):
                self.board[j][y] = new_col[j]
        
        self.score += new_score
        return new_score
    
    def move_down(self) -> int:
        """
        Moves tiles down and merges like tiles.

        :return Points: the # of points added to the total score from making this move.
        """
        new_score = 0

        for y in range(self.size):
            col_values = [row[y] for row in self.board if row[y] != 0]
            i = len(col_values) - 1

            new_col = [0 for _ in range(self.size)]
            new_index = self.size - 1

            while i >= 0:
                if i > 0 and col_values[i] == col_values[i - 1]:
                    new_col[new_index] = col_values[i] * 2
                    new_score += new_col[new_index]
                    new_index -= 1
                    i -= 2
                else:
                    new_col[new_index] = col_values[i]
                    new_index -= 1
                    i -= 1
            
            for j in range(self.size):
                self.board[j][y] = new_col[j]

        self.score += new_score
        return new_score
    
    def play(self) -> None:
        """
        Run an interactive terminal game.
        """
        while not self.is_game_over():
            self.print()

            legal_moves = self.get_legal_moves()
            move = input("Enter move (w, a, s, d): ")

            if move == "a" and "LEFT" in legal_moves:
                self.move_left()
            elif move == "d" and "RIGHT" in legal_moves:
                self.move_right()
            elif move == "w" and "UP" in legal_moves:
                self.move_up()
            elif move == "s" and "DOWN" in legal_moves:
                self.move_down()
            else:
                continue

            self.spawn_tile()
        
        self.print()
        print("Game over!")


if __name__ == "__main__":
    game = Game()
    game.play()