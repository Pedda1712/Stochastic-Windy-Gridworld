from Agent import SarsaAgent, RecordingAgent, QLearningAgent
from Transition import WindyTransition, WindyKingTransition, StochasticWindyKingTransition
from Environment import Environment

# visualize training process 
repeats = 200
episodes = 200
length = [(0,0)] * episodes
trajectory = None
agent = None
for r in range(repeats):
    print(f"{int(r*100/repeats)}%")
    goal = (7,3)
    #transition = WindyTransition(goal_state = goal)
    #transition = WindyKingTransition(goal_state = goal)
    transition = StochasticWindyKingTransition(goal_state = goal)
    agent = RecordingAgent(QLearningAgent(0.1, 0.5, transition, goal))
    #agent = RecordingAgent(SarsaAgent(0.1, 0.5, transition, goal))
    env = Environment(
        transition,
        [agent],
        0,
        lambda seed: (0, 3),
        lambda seed: 0)
    for ep in range(episodes):
        print(f"{r}<-{int(100*ep/episodes)}%", end="\r")
        env.init_episode(ep)
        count = 0
        while True:
            env.step()
            if agent.is_done():
                break
            count += 1
            c, a = length[ep]
            c += 1
            length[ep] = (c, a + (1/c)*(count - a))
    print("")

import matplotlib.pyplot as plt
length = [l for (_, l) in length]
plt.plot(length)
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Length of Episode')
plt.show()

# train agent for trajectory visualisation
goal = (7,3)
#transition = WindyTransition(goal_state = goal)
#transition = WindyKingTransition(goal_state = goal)
transition = StochasticWindyKingTransition(goal_state = goal)
agent = RecordingAgent(QLearningAgent(0.01, 0.5, transition, goal))
#agent = RecordingAgent(SarsaAgent(0.1, 0.5, transition, goal))
env = Environment(
    transition,
    [agent],
    0,
    lambda seed: (0, 3),
    lambda seed: 0)
for ep in range(5000):
    env.init_episode(ep)
    while True:
        env.step()
        if agent.is_done():
            break

# only take greedy actions in 'deployment'
agent.agent.set_greedy(True)
while True:
    env.init_episode(0)
    while True:
        env.step()
        if agent.is_done():
            break
    trajectory = agent.get_recording()
    arr = [[0 for _ in range(10)] for _ in range(7)]
    for (i, (x,y)) in enumerate(trajectory):
        arr[6- y][x] = i
    plt.matshow(arr)
    plt.xlabel('X')
    plt.xlabel('Y')
    plt.title('Final Trajectory')
    plt.show()
    #break
