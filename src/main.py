import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import os

from DDQN import DoubleDQN
from GridWorld import GridWorld

os.makedirs('plots', exist_ok=True)
os.makedirs('agents', exist_ok=True)

random.seed(111)
np.random.seed(111)
tf.random.set_seed(111)


def plot_performance(episodes, scores, losses, epsilons, window_size=100, filename='performance_plot.png'):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    if len(scores) >= window_size:
        # rolling averages
        rolling_scores = np.convolve(scores, np.ones(window_size), 'valid') / window_size
        rolling_losses = np.convolve(losses, np.ones(window_size), 'valid') / window_size

        x_axis = range(window_size - 1, len(episodes))
    else:
        rolling_scores = scores
        rolling_losses = losses
        x_axis = episodes

    # Average reward
    ax1.plot(x_axis, rolling_scores, label='Rolling Average')
    ax1.plot(episodes, scores, alpha=0.3, label='Raw Scores')
    ax1.set_title(f'Average Reward (Rolling Window: {window_size})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()

    # Average loss
    ax2.plot(x_axis, rolling_losses, label='Rolling Average', color='orange')
    ax2.plot(episodes, losses, alpha=0.3, label='Raw Losses', color='orange')
    ax2.set_title(f'Average Loss (Rolling Window: {window_size})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # Epsilon
    ax3.plot(episodes, epsilons, color='green')
    ax3.set_title('Epsilon Value Over Time')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.legend(['Epsilon'])

    plt.tight_layout()
    plt.savefig(f'plots/{filename}')  # Save plot into plots/ dir
    plt.close()

def plot_path(grid, path, goal, episode, filename='path_plot.png'):
    plt.imshow(grid, cmap='gray', origin='upper')

    # Color gradient
    cmap = plt.colormaps['autumn']
    path = np.array(path)

    for i in range(len(path) - 1):
        plt.plot([path[i, 1], path[i + 1, 1]], [path[i, 0], path[i + 1, 0]],
                 color=cmap(i / len(path)), lw=2)  # Gradient color

    plt.scatter([goal[1]], [goal[0]], color='blue', s=100, label='Goal')  # Goal marker
    plt.title(f'Path taken by the agent - Episode {episode}')
    plt.legend()

    plt.savefig(f'plots/{filename}')  # Save plot into plots/ dir
    plt.close()  # close after saving

def plot_rolling_loss(episodes, losses, window_size=100, filename='rolling_loss_plot.png'):
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(losses) >= window_size:
        # rolling average
        rolling_losses = np.convolve(losses, np.ones(window_size), 'valid') / window_size
        x_axis = range(window_size - 1, len(episodes))
    else:
        rolling_losses = losses
        x_axis = episodes

    # Rolling average loss
    ax.plot(x_axis, rolling_losses, label='Rolling Average Loss', color='orange')
    ax.set_title(f'Rolling Average Loss (Window: {window_size})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'plots/{filename}')  # Save to plots/
    plt.close()


def save_and_continue(agent: DoubleDQN, error_message=None, episode=0):
    try:
        agent.model.save_weights(f'agents/agent_main_model_{episode}.weights.h5')
        agent.target_model.save_weights(f'agents/agent_target_model_{episode}.weights.h5')
        print("Models saved successfully.")
    except Exception as e:
        print(f"Error saving models: {e}")

    if error_message:
        print(f"Error: {error_message}")
        print("Attempting to continue training...")

# Training -------------------------------------------------------------------------------------------------------------
env = GridWorld(size=10)
state_shape = (10, 10, 3)
action_size = 5
agent = DoubleDQN(state_shape, action_size)
batch_size = 32
episodes = 30000
window_size = 300

visualiser_update_frequency = 500
model_save_frequency = 3000
target_model_update_frequency = 10

scores = []
losses = []
epsilons = []

for e in range(episodes):
    state = env.reset()
    goal = env.goal.copy()
    score = 0
    path = []  # save path

    for time in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        path.append(env.position.copy())

        if done:
            break

    scores.append(score)
    epsilons.append(agent.epsilon)

    if len(agent.memory) > batch_size:
        loss = agent.replay(batch_size)
        losses.append(loss)
    else:
        losses.append(0)

    if e % target_model_update_frequency == 0:
        agent.update_target_model()

    if e % visualiser_update_frequency == 0:
        print(f"Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")

        # Path
        try:
            plot_path(env.grid, path, goal, e, filename=f'path_plot_episode_{e}.png')
        except Exception as plot_error:
            print(f"Plot path error: {plot_error}")
            save_and_continue(agent, f"Failed to plot path on episode {e}", episode=e)

        # Average Reward, Loss, Epsilon
        try:
            plot_performance(range(e + 1), scores, losses, epsilons, window_size,
                             filename=f'performance_plot_episode_{e}.png')
        except Exception as plot_error:
            print(f"Performance plot error: {plot_error}")
            save_and_continue(agent, f"Failed to plot performance on episode {e}", episode=e)

        # Rolling average exclusive graph
        try:
            plot_rolling_loss(range(e + 1), losses, window_size, filename=f'rolling_loss_plot_episode_{e}.png')
        except Exception as plot_error:
            print(f"Rolling loss plot error: {plot_error}")
            save_and_continue(agent, f"Failed to plot rolling loss on episode {e}", episode=e)

    if e % model_save_frequency == 0:
        try:
            agent.model.save_weights(f'agents/agent_main_model_{e}.weights.h5')
            agent.target_model.save_weights(f'agents/agent_target_model_{e}.weights.h5')
        except Exception as save_error:
            print(f"Save error: {save_error}")


# final plot
plot_performance(range(episodes), scores, losses, epsilons, window_size, filename=f'performance_plot_episode_{episodes}.png')
plot_rolling_loss(range(episodes), losses, window_size, filename=f'rolling_loss_plot_episode_{episodes}.png')
agent.model.save_weights(f'agents/agent_main_model_{episodes}.weights.h5')
agent.target_model.save_weights(f'agents/agent_target_model_{episodes}.weights.h5')
