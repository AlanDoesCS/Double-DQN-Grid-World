import numpy as np
import tensorflow as tf
from GridWorld import GridWorld
from DDQN import DoubleDQN
import heapq
import random
import os
from collections import deque
from tqdm import tqdm

np.random.seed(111)
random.seed(111)
tf.random.set_seed(111)


def dijkstra(grid, start, goal):
    size = grid.shape[0]
    visited = np.full((size, size), False)
    distances = np.full((size, size), np.inf)
    distances[start[0], start[1]] = 0
    prev = np.empty((size, size), dtype=object)

    heap = [(0, start)]
    nodes_visited = 0

    while heap:
        dist, current = heapq.heappop(heap)
        if visited[current[0], current[1]]:
            continue
        visited[current[0], current[1]] = True
        nodes_visited += 1

        if current == goal:
            break

        neighbors = []
        if current[0] > 0:
            neighbors.append((current[0] - 1, current[1]))
        if current[0] < size - 1:
            neighbors.append((current[0] + 1, current[1]))
        if current[1] > 0:
            neighbors.append((current[0], current[1] - 1))
        if current[1] < size - 1:
            neighbors.append((current[0], current[1] + 1))

        for neighbor in neighbors:
            if not visited[neighbor[0], neighbor[1]]:
                alt = dist + grid[neighbor[0], neighbor[1]]
                if alt < distances[neighbor[0], neighbor[1]]:
                    distances[neighbor[0], neighbor[1]] = alt
                    prev[neighbor[0], neighbor[1]] = current
                    heapq.heappush(heap, (alt, neighbor))

    path = []
    current = goal
    if distances[goal[0], goal[1]] < np.inf:
        while current != start:
            path.append(current)
            current = prev[current[0], current[1]]
        path.append(start)
        path = path[::-1]
    else:
        path = None  # No path found

    return path, distances[goal[0], goal[1]], nodes_visited


def astar(grid, start, goal):
    size = grid.shape[0]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = np.empty((size, size), dtype=object)
    g_score = np.full((size, size), np.inf)
    f_score = np.full((size, size), np.inf)
    g_score[start[0], start[1]] = 0
    f_score[start[0], start[1]] = heuristic(start, goal)
    nodes_visited = 0
    visited = np.full((size, size), False)

    while open_set:
        _, current = heapq.heappop(open_set)
        if visited[current[0], current[1]]:
            continue
        visited[current[0], current[1]] = True
        nodes_visited += 1

        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current[0], current[1]]
            path.append(start)
            path = path[::-1]
            total_weight = g_score[goal[0], goal[1]]
            return path, total_weight, nodes_visited

        neighbors = []
        if current[0] > 0:
            neighbors.append((current[0] - 1, current[1]))
        if current[0] < size - 1:
            neighbors.append((current[0] + 1, current[1]))
        if current[1] > 0:
            neighbors.append((current[0], current[1] - 1))
        if current[1] < size - 1:
            neighbors.append((current[0], current[1] + 1))

        for neighbor in neighbors:
            tentative_g_score = g_score[current[0], current[1]] + grid[neighbor[0], neighbor[1]]
            if tentative_g_score < g_score[neighbor[0], neighbor[1]]:
                came_from[neighbor[0], neighbor[1]] = current
                g_score[neighbor[0], neighbor[1]] = tentative_g_score
                f_score[neighbor[0], neighbor[1]] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor[0], neighbor[1]], neighbor))

    return None, np.inf, nodes_visited  # No path found


def heuristic(a, b):  # Calculate manhattan distance between a and b
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs(grid, start, goal):
    size = grid.shape[0]
    queue = deque()
    queue.append(start)
    visited = np.full((size, size), False)
    visited[start[0], start[1]] = True
    came_from = np.empty((size, size), dtype=object)
    nodes_visited = 1

    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current[0], current[1]]
            path.append(start)
            path = path[::-1]

            total_weight = sum([grid[pos[0], pos[1]] for pos in path[1:]])
            return path, total_weight, nodes_visited

        neighbors = []
        if current[0] > 0:
            neighbors.append((current[0] - 1, current[1]))
        if current[0] < size - 1:
            neighbors.append((current[0] + 1, current[1]))
        if current[1] > 0:
            neighbors.append((current[0], current[1] - 1))
        if current[1] < size - 1:
            neighbors.append((current[0], current[1] + 1))

        for neighbor in neighbors:
            if not visited[neighbor[0], neighbor[1]]:
                visited[neighbor[0], neighbor[1]] = True
                came_from[neighbor[0], neighbor[1]] = current
                queue.append(neighbor)
                nodes_visited += 1

    return None, np.inf, nodes_visited  # No path found


def evaluate_model(agent, num_trials=500, max_steps=200):
    success_count = 0
    total_steps = []
    total_path_weights = []
    total_visited_nodes = []
    accuracies = []
    losses = []

    for _trial in tqdm(range(num_trials), desc="Evaluating DDQN", unit="trial"):
        env = GridWorld(size=10)
        state = env.reset()
        start = env.position.copy()
        goal = env.goal.copy()

        grid = env.grid.copy()

        dijkstra_path, dijkstra_weight, _ = dijkstra(grid, tuple(start), tuple(goal))

        if dijkstra_path is None:
            continue

        # Run the agent
        agent_state = state.copy()
        agent_env = GridWorld(size=10)
        agent_env.grid = grid.copy()
        agent_env.position = start.copy()
        agent_env.goal = goal.copy()

        agent_path = [agent_env.position.copy()]
        steps = 0
        visited_positions = set()
        done = False
        while not done and steps < max_steps:
            action = agent.act(agent_state)
            next_state, reward, done = agent_env.step(action)
            agent_state = next_state.copy()
            agent_position = agent_env.position.copy()
            agent_path.append(agent_position)
            visited_positions.add(tuple(agent_position))
            steps += 1

        if done:
            success_count += 1

        total_steps.append(steps)
        # Compute λ(PA)
        agent_path_weight = 0.0
        for pos in agent_path[1:]:
            agent_path_weight += grid[pos[0], pos[1]]

        total_path_weights.append(agent_path_weight)
        total_visited_nodes.append(len(visited_positions))

        # Compute accuracy
        accuracy = (1 - abs(dijkstra_weight - agent_path_weight) / dijkstra_weight) * 100
        accuracies.append(accuracy)

    mean_steps = np.mean(total_steps)
    mean_path_weight = np.mean(total_path_weights)
    mean_visited_nodes = np.mean(total_visited_nodes)
    mean_accuracy = np.mean(accuracies)
    success_rate = success_count / num_trials * 100
    mean_loss = np.nan

    return {
        'Mean Visited Node Count': mean_visited_nodes,
        'Mean Path Length': mean_steps,
        'Accuracy (%)': mean_accuracy,
        'Success Rate (%)': success_rate,
        'Mean average loss': mean_loss
    }


def evaluate_dijkstra(num_trials=500):
    total_steps = []
    total_path_weights = []
    total_visited_nodes = []
    accuracies = []
    success_count = 0

    for trial in range(num_trials):
        env = GridWorld(size=10)
        state = env.reset()
        start = env.position.copy()
        goal = env.goal.copy()

        grid = env.grid.copy()

        dijkstra_path, dijkstra_weight, nodes_visited = dijkstra(grid, tuple(start), tuple(goal))

        if dijkstra_path is None:
            continue

        path_length = len(dijkstra_path) - 1

        success_count += 1
        total_steps.append(path_length)
        total_path_weights.append(dijkstra_weight)
        total_visited_nodes.append(nodes_visited)
        accuracies.append(100.0)  # Already known to be 100% accurate

    # Compute mean values
    mean_steps = np.mean(total_steps)
    mean_path_weight = np.mean(total_path_weights)
    mean_visited_nodes = np.mean(total_visited_nodes)
    mean_accuracy = np.mean(accuracies)
    success_rate = success_count / num_trials * 100
    mean_loss = np.nan

    return {
        'Mean Visited Node Count': mean_visited_nodes,
        'Mean Path Length': mean_steps,
        'Accuracy (%)': mean_accuracy,
        'Success Rate (%)': success_rate,
        'Mean average loss': mean_loss
    }


def evaluate_astar(num_trials=500):
    total_steps = []
    total_path_weights = []
    total_visited_nodes = []
    accuracies = []
    success_count = 0

    for trial in range(num_trials):
        env = GridWorld(size=10)
        state = env.reset()
        start = env.position.copy()
        goal = env.goal.copy()

        grid = env.grid.copy()

        dijkstra_path, dijkstra_weight, _ = dijkstra(grid, tuple(start), tuple(goal))

        if dijkstra_path is None:
            continue

        astar_path, astar_weight, nodes_visited = astar(grid, tuple(start), tuple(goal))

        if astar_path is None:
            continue

        path_length = len(astar_path) - 1

        success_count += 1
        total_steps.append(path_length)
        total_path_weights.append(astar_weight)
        total_visited_nodes.append(nodes_visited)

        accuracy = (1 - abs(dijkstra_weight - astar_weight) / dijkstra_weight) * 100
        accuracies.append(accuracy)

    mean_steps = np.mean(total_steps)
    mean_path_weight = np.mean(total_path_weights)
    mean_visited_nodes = np.mean(total_visited_nodes)
    mean_accuracy = np.mean(accuracies)
    success_rate = success_count / num_trials * 100
    mean_loss = np.nan

    return {
        'Mean Visited Node Count': mean_visited_nodes,
        'Mean Path Length': mean_steps,
        'Accuracy (%)': mean_accuracy,
        'Success Rate (%)': success_rate,
        'Mean average loss': mean_loss
    }


def evaluate_bfs(num_trials=500):
    total_steps = []
    total_path_weights = []
    total_visited_nodes = []
    accuracies = []
    success_count = 0

    for trial in range(num_trials):
        env = GridWorld(size=10)
        state = env.reset()
        start = env.position.copy()
        goal = env.goal.copy()

        grid = env.grid.copy()

        # Dijkstra, used for accuracy calculation
        dijkstra_path, dijkstra_weight, _ = dijkstra(grid, tuple(start), tuple(goal))

        if dijkstra_path is None:
            # No path found by Dijkstra
            print("WARNING: No path found by Dijkstra")
            continue

        bfs_path, bfs_weight, nodes_visited = bfs(grid, tuple(start), tuple(goal))

        if bfs_path is None:
            # No path found
            print("WARNING: No path found by BFS")
            continue

        path_length = len(bfs_path) - 1

        success_count += 1
        total_steps.append(path_length)
        total_path_weights.append(bfs_weight)
        total_visited_nodes.append(nodes_visited)

        accuracy = (1 - abs(dijkstra_weight - bfs_weight) / dijkstra_weight) * 100
        accuracies.append(accuracy)

    # Compute mean values
    mean_steps = np.mean(total_steps)
    mean_path_weight = np.mean(total_path_weights)
    mean_visited_nodes = np.mean(total_visited_nodes)
    mean_accuracy = np.mean(accuracies)
    success_rate = success_count / num_trials * 100
    mean_loss = np.nan

    return {
        'Mean Visited Node Count': mean_visited_nodes,
        'Mean Path Length': mean_steps,
        'Accuracy (%)': mean_accuracy,
        'Success Rate (%)': success_rate,
        'Mean average loss': mean_loss
    }


def main():
    episodes_list = list(range(0, 30001, 3000))
    results = {}

    state_shape = (10, 10, 3)
    action_size = 5

    print("Evaluating Dijkstra's algorithm")
    dijkstra_metrics = evaluate_dijkstra(num_trials=500)
    results['Dijkstra'] = dijkstra_metrics

    print("Evaluating A* algorithm")
    astar_metrics = evaluate_astar(num_trials=500)
    results['A*'] = astar_metrics

    print("Evaluating BFS algorithm")
    bfs_metrics = evaluate_bfs(num_trials=500)
    results['BFS'] = bfs_metrics

    for episode in episodes_list:
        print(f"Evaluating model at episode {episode}")
        # Load the agent
        agent = DoubleDQN(state_shape, action_size)
        # Load weights
        try:
            agent.model.load_weights(f'agents/agent_main_model_{episode}.weights.h5')
            agent.target_model.load_weights(f'agents/agent_target_model_{episode}.weights.h5')
            agent.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=agent.learning_rate))
            agent.target_model.compile(loss='mse',
                                       optimizer=tf.keras.optimizers.Adam(learning_rate=agent.learning_rate))
        except Exception as e:
            print(f"Could not load weights for episode {episode}: {e}")
            continue

        # No exploration
        agent.epsilon = 0

        metrics = evaluate_model(agent, num_trials=500, max_steps=200)
        results[f'DDQN ({episode} episodes)'] = metrics

    # Results output
    for algo, metrics in results.items():
        print(f"Algorithm: {algo}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print()


if __name__ == '__main__':
    main()
