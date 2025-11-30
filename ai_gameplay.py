from game import Game
from expectimax_agent import ExpectimaxAgent
#import time


def play_ai():
    print("AI starting...")
    game = Game()
    agent = ExpectimaxAgent(depth=2)   #depth can be changed. smaller num = faster

    while not game.is_game_over():
        game.print()
        move = agent.get_best_move(game)
        print("AI Move:", move)

        if move == "LEFT":
            game.move_left()
        elif move == "RIGHT":
            game.move_right()
        elif move == "UP":
            game.move_up()
        elif move == "DOWN":
            game.move_down()
        else:
            print("NO VALID MOVE")
            break

        game.spawn_tile()
        #time.sleep(0.25)

    print("AI finished:")
    game.print()

if __name__ == "__main__":
    play_ai()
