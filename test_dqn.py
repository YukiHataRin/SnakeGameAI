import argparse
import torch
from dqn.agent import Agent
from dqn.game_wrapper_mlp import SnakeGameAI

def test(model_path, height, width, speed, games):
    # 加載模型
    agent = Agent(pretrain_model=model_path)
    game = SnakeGameAI(h = height, w = width, speed = speed)

    for g in range(games):
        print(f"Game: {g + 1} ", end="")
        # 遊戲循環
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action_eval(state_old)
            reward, done, score = game.play_step(final_move)
            if done:
                print(f'遊戲結束，得分：{score}')
                break
            
        game.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a DQN model on Snake Game")
    parser.add_argument("--model_path", type=str, help="Path to the trained model")
    parser.add_argument("--height", type=int, default=12, help="Number of vertical grids")
    parser.add_argument("--width", type=int, default=12, help="Number of horizontal grids")
    parser.add_argument("--speed", type=int, default=30, help="Speed of the game")
    parser.add_argument("--games", type=int, default=1, help="Number of the game plays")

    args = parser.parse_args()
    test(args.model_path, args.height * 20, args.width * 20, args.speed, args.games)
