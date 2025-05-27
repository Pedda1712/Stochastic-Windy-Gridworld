"""
Modular Environment Interfaces.
"""
from Agent import BaseAgent
from Transition import ITransition
from typing import TypeVar, Generic, Callable

S = TypeVar('S')
A = TypeVar('A')

class Environment(Generic[S]):
    transition: ITransition
    agents: list[BaseAgent]

    def __init__(self, transition: ITransition, agents: list[BaseAgent], seed: int, initial_state_generator: Callable[[int], S], initial_reward_generator: Callable[[int], float]):
        self.transition = transition
        self.agents = agents
        self.initial_state_generator = initial_state_generator
        self.initial_reward_generator = initial_reward_generator
        self.init_episode(seed)

    def init_episode(self, seed):
        for a in self.agents:
            a.reset()
        # generate initial states
        self.last_states_and_rewards = [
            (self.initial_state_generator(seed + i), self.initial_reward_generator(seed + i))
            for (i, _) in enumerate(self.agents)
        ]
    def step(self):
        self.last_states_and_rewards = self.transition.transition(
            [(state, agent.observe(reward, state))
             for (agent, (state, reward)) in
             zip(self.agents, self.last_states_and_rewards)])
