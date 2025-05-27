"""
Agent wrapper that records trajectory.
"""
from typing import Generic
from .agent import S, A, BaseAgent

class RecordingAgent(Generic[S,A], BaseAgent[S,A]):
    trajectory: list[S]
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.trajectory = []

    def get_recording(self):
        return self.trajectory

    def is_done(self):
        return self.agent.is_done()

    def reset(self):
        self.trajectory = []
        self.agent.reset()

    def get_state(self):
        return self.agent.get_state()

    def observe(self, reward, state):
        self.trajectory.append(state)
        return self.agent.observe(reward, state)
