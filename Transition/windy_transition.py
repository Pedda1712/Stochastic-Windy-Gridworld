"""
Transition function for the windy gridworld.
"""
from .transition import ITransition
from random import random

class WindyTransition(ITransition[tuple[int, int], tuple[int, int]]):
    """Transition for the Windy gridworld"""
    def __init__(self, winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], height = 7, goal_state = (7, 3)):
        self.winds = winds
        self.height = height
        self.width = len(self.winds)
        self.goal = goal_state

    def get_permissible(self, state: tuple[int, int]) -> list[tuple[int,int]]:
        return [(1,0), (0,1), (-1,0), (0,-1)]
        
    def _transition_one(self, state: tuple[int, int], action: tuple[int, int]) -> tuple[tuple[int, int], float]:

        if state == self.goal: # unified notation for episodic and continuing tasks
            return (self.goal, 0)
        
        action = (action[0], action[1] + self.winds[state[0]])
        next_state = (
            min(self.width - 1, max(0, state[0] + action[0])),
            min(self.height - 1, max(0, state[1] + action[1]))
        )

        if next_state == self.goal:
            return (self.goal, 0)
        return (next_state, -1)

        
    def transition(self, states_and_actions: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[tuple[tuple[int, int], float]]:
        return [self._transition_one(state, action) for (state, action) in states_and_actions]
    
class WindyKingTransition(WindyTransition):

        def __init__(self, winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], height = 7, goal_state = (7, 3)):
            super().__init__(winds, height, goal_state)

        def get_permissible(self, state: tuple[int, int]) -> list[tuple[int,int]]:
            return [(1,0), (0,1), (-1,0), (0,-1), (0, 0), (1,1), (-1,1), (-1, -1), (1, -1)]

class StochasticWindyKingTransition(WindyKingTransition):
    
        def __init__(self, winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], height = 7, goal_state = (7, 3)):
            super().__init__(winds, height, goal_state)

        def _transition_one(self, state: tuple[int, int], action: tuple[int, int]) -> tuple[tuple[int, int], float]:
                
            if state == self.goal: 
                return (self.goal, 0)

            rng = int(random() * 3) - 1 # the only difference for the stochastic wind
        
            action = (action[0], action[1] + self.winds[state[0]] + rng)
            next_state = (
                min(self.width - 1, max(0, state[0] + action[0])),
                min(self.height - 1, max(0, state[1] + action[1]))
            )

            if next_state == self.goal:
                return (self.goal, 0)
            return (next_state, -1)
