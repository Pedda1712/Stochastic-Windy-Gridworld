"""
QLearning Agent Implementation.
"""
from Transition import ITransition
from typing import TypeVar, Generic, Optional
from .agent import S, A
from .sarsa_agent import SarsaAgent
import random

class QLearningAgent(Generic[S,A], SarsaAgent[S,A]):

    def __init__(self, epsilon: float, step_size: float, transition: ITransition, goal: S, discount: float = 1, initial_Q: float = 0):
        super().__init__(epsilon, step_size, transition, goal, discount, initial_Q)

    def update_q_values(self, state: S, action: A):
        permissible_actions = self.transition.get_permissible(state)
        q_values = [self._Q(state, a) for a in permissible_actions]
        greedy = permissible_actions[q_values.index(max(q_values))]
        
        self.Q[(self.state, self.action)] = self._Q(self.state, self.action) + self.step_size * (self.reward + self.discount * self._Q(state, greedy) - self._Q(self.state, self.action))
