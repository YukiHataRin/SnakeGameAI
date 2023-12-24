import argparse
from dqn.agent import Agent
from dqn.game_wrapper_mlp import SnakeGameAI
from dqn.model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt

def train(epochs, height, width, speed, pretrain_model=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI(w=width, h=height, speed=speed)
    agent = Agent(pretrain_model=pretrain_model)

    while agent.n_games < epochs:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record or record == 0:
                record = score
                agent.model.save()

            print(f'Game: {agent.n_games} Score: {score} Highest: {record} ')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            print(f'Mean: {mean_score}')

    # 繪製圖表
    plt.figure(figsize=(10, 5))  # 設置圖表大小
    plt.plot(plot_scores, label='Score')
    plt.plot(plot_mean_scores, label='Mean Score', linestyle='--')
    plt.title('Game-Score')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # 保存圖表
    if pretrain_model:
        plt.savefig(f'res/game_scores_{pretrain_model}.png')

    else:
        plt.savefig(f'res/game_scores_dqn_model.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Snake")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--height", type=int, default=12, help="Number of vertical grids")
    parser.add_argument("--width", type=int, default=12, help="Number of horizontal grids")
    parser.add_argument("--speed", type=int, default=100, help="Speed of the game")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to a pretrained model")

    args = parser.parse_args()

    train(args.epochs, args.height * 20, args.width * 20, args.speed, args.pretrain)
