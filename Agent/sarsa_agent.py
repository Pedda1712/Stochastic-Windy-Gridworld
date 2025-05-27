"""
Sarsa Agent Implementation.
"""
from Transition import ITransition
from typing import TypeVar, Generic, Optional
from .agent import S, A, BaseAgent
import random

class SarsaAgent(Generic[S,A], BaseAgent[S,A]):
    epsilon: float
    step_size: float
    initial_Q: float

    # the 'SAR' part of 'SARSA' :-)
    state: Optional[S]
    action: Optional[A]
    reward: Optional[float]

    Q: dict[tuple[S, A], float]

    def __init__(self, epsilon: float, step_size: float, transition: ITransition, goal: S, discount: float = 1, initial_Q: float = 0):
        self.epsilon = epsilon
        self.step_size = step_size
        self.transition = transition
        self.state = None
        self.action = None
        self.reward = None
        self.initial_Q = initial_Q
        self.Q = {}
        self.goal = goal
        self.discount = discount
        self.greedy = False

    def reset(self):
        self.state = None
        self.action = None
        self.reward = None

    def is_done(self):
        return self.state == self.goal if self.state else False
        
    def get_state(self) -> Optional[S]:
        return self.state

    def _Q(self, state, action):
        if state == self.goal:
            return 0
        if (state, action) in self.Q:
            return self.Q[(state, action)]
        return self.initial_Q

    def set_greedy(self, g: bool):
        self.greedy = g
    
    def choose(self, state):
        # epsilon-greedy action selection
        permissible_actions = self.transition.get_permissible(state)
        q_values = [self._Q(state, a) for a in permissible_actions]

        N = len(q_values)
        greedy = q_values.index(max(q_values))
        if self.greedy:
            return permissible_actions[greedy]
        v = random.random()
                    
        if v < self.epsilon:
            return permissible_actions[int(v / self.epsilon)]
        return permissible_actions[greedy]

    def update_q_values(self, state: S, action: A):
        self.Q[(self.state, self.action)] = self._Q(self.state, self.action) + self.step_size * (self.reward + self.discount * self._Q(state, action) - self._Q(self.state, self.action))
                    
    def observe(self, reward: float, state: S) -> A:
        self.reward = reward # this is the reward of the LAST time step

        action = self.choose(state)

        self.update_q_values(state, action)

        self.state = state
        self.action = action

        return action

