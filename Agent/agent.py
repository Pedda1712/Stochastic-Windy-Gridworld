"""
Agent Interface.
"""
from typing import TypeVar, Generic, Optional
from Transition import ITransition

S = TypeVar('S')
A = TypeVar('A')

class BaseAgent(Generic[S, A]):
    state: Optional[S]
    default_action = A
    
    def __init__(self, default_action: A, goal: S):
        self.state = None
        self.goal = goal
        self.default_action = default_action

    def is_done(self):
        return self.state == self.goal if self.state else False
        
    def reset(self):
        """Reset learner for next episode."""
        self.state = None
        
    def get_state(self) -> Optional[S]:
        """Return last observed state."""
        return self.state

    def observe(self, reward: float, state: S) -> A:
        """
        Let the agent observe a reward and a state.

        Reward is the reward for the previous action taken, so essentially:
           observe(R_t-1, S_t, A_t)
        An agent should internally keep track of the required rewards
        and update its policy.

        Parameters:
          reward: reward obtained for the previous action
          state: current state

        Returns:
          what action to take in S
        """
        self.state = state
        return self.default_action
