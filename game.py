import random
import copy


class Game:
    """
    The 2048 game class
    """

    def __init__(self, size = 4):
        self.size = size
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.score = 0

    def print(self):
        """
        Print the board in a formatted grid.
        """
        board_width = self.size * 5 + 1

        # Score header
        print("+" + "-" * board_width + "+")
        print(f"| Score: {self.score:<{board_width - 8}}" + "|")
        print("+" + "-" * board_width + "+")

        for row in self.board:
            print("|", end="")                              # left border

            for tile in row:
                if tile == 0:
                    print("{:>5}".format("."), end="")      # do not print zeros
                else:
                    print("{:>5}".format(tile), end="")
            
            print(" |")                                     # right border and new line
        
        print("+" + "-" * board_width + "+")        # bottom border
    
    def get_empty_tiles(self):
        """
        Returns x, y positions of empty tiles.
        """
        empty_tiles = []

        for x in range( len(self.board) ):
            for y in range( len(self.board[x]) ):
                if self.board[x][y] == 0:
                    empty_tiles.append( (x, y) )
        
        return empty_tiles
    
    def spawn_tile(self):
        """
        Adds a 2 or 4 tile randomly to the board.
        """
        new_tile = 2 if random.random() < 0.9 else 4        # 90% chance for 2 tile, 10% for 4 tile

        empty_tiles = self.get_empty_tiles()

        if empty_tiles:
            x, y = random.choice(empty_tiles)               # pick a random empty tile
            self.board[x][y] = new_tile
        else:
            raise ValueError("Cannot add new tile: board is full.")


    def is_game_over(self):
        """
        Returns true if there are no more available moves.
        """

        if self.get_legal_moves():
            return False
        
        # There are no remaining moves
        return True
    
    def get_legal_moves(self):
        """
        :return: Set of legal moves
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
    
    def move_left(self):
        """
        Moves tiles to the left and merges like tiles.
        """

        for x, row in enumerate(self.board):
            row_values = [n for n in row if n != 0]     # the current row, excluding empty tiles
            tile_index = 0                              # index of the tile being checked

            new_row = [0 for _ in range(self.size)]     # the new, merged row
            new_index = 0                               # index of the next tile to fill

            while tile_index < len(row_values):
                if (
                    tile_index < ( len(row_values) - 1) and
                    row_values[tile_index] == row_values[tile_index + 1]
                ):
                    new_row[new_index] = row_values[tile_index] * 2
                    self.score += new_row[new_index]
                    new_index += 1
                    tile_index += 2
                else:
                    new_row[new_index] = row_values[tile_index]
                    new_index += 1
                    tile_index += 1
            
            self.board[x] = new_row     # insert the new row
    
    def move_right(self):
        """
        Moves tiles to the right and merges like tiles.
        """

        for x, row in enumerate(self.board):
            row_values = [n for n in row if n != 0]     # the current row, excluding empty tiles
            tile_index = len(row_values) - 1            # index of the tile being checked
            
            new_row = [0 for _ in range(self.size)]     # the new, merged row
            new_index = self.size - 1                   # index of the next tile to fill

            while tile_index >= 0:
                if (
                    tile_index > 0 and
                    row_values[tile_index] == row_values[tile_index - 1]
                ):
                    new_row[new_index] = row_values[tile_index] * 2
                    self.score += new_row[new_index]
                    new_index -= 1
                    tile_index -= 2
                else:
                    new_row[new_index] = row_values[tile_index]
                    new_index -= 1
                    tile_index -= 1
            
            self.board[x] = new_row     # insert the new row
    
    def move_up(self):
        """
        Moves tiles up and merges like tiles.
        """

        for y in range(self.size):
            col_values = [
                row[y] for row in self.board if row[y] != 0     # the current column, excluding empty tiles
            ]
            tile_index = 0                                      # index of the tile being checked

            new_col = [0 for _ in range(self.size)]             # the new, merged column
            new_index = 0                                       # index of the next tile to fill

            while tile_index < len(col_values):
                if (
                    tile_index < ( len(col_values) - 1) and
                    col_values[tile_index] == col_values[tile_index + 1]
                ):
                    new_col[new_index] = col_values[tile_index] * 2
                    self.score += new_col[new_index]
                    new_index += 1
                    tile_index += 2
                else:
                    new_col[new_index] = col_values[tile_index]
                    new_index += 1
                    tile_index += 1
            
            for j in range(self.size):              # insert the new column
                self.board[j][y] = new_col[j]
    
    def move_down(self):
        """
        Moves tiles down and merges like tiles.
        """

        for y in range(self.size):
            col_values = [row[y] for row in self.board if row[y] != 0]
            i = len(col_values) - 1                     # index of the tile being checked

            new_col = [0 for _ in range(self.size)]    # the new, merged column
            new_index = self.size - 1                  # index of the next tile to fill

            while i >= 0:
                if i > 0 and col_values[i] == col_values[i - 1]:
                    new_col[new_index] = col_values[i] * 2
                    self.score += new_col[new_index]
                    new_index -= 1
                    i -= 2
                else:
                    new_col[new_index] = col_values[i]
                    new_index -= 1
                    i -= 1
            
            for j in range(self.size):              # insert the new column
                self.board[j][y] = new_col[j]
    
    def play(self):
        """
        Play the game
        """

        # Start with 2 tiles
        self.spawn_tile()
        self.spawn_tile()

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