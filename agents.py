import random
import numpy as np
from math import sqrt, exp

from settings import *


class DynaQ:
    def __init__(self,
                 action_space=(0, 1, 2, 3),
                 state_shape=(6, 9),
                 epsilon_final=0.1,
                 epsilon_start=0.1,
                 alpha=0.1,
                 gamma=0.95,
                 n=50,
                 k=1e-4,
                 ):

        self.action_space = action_space
        self.action_space_size = len(action_space)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.eps_interval = epsilon_start - epsilon_final
        self.eps_decay_factor = (epsilon_final/epsilon_start)**(1/MAX_TIMESTEP)

        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.k = k

        self.timestep = 0

        q_shape = list(state_shape)
        q_shape.append(len(action_space))
        self.q_table = np.zeros(q_shape, dtype='float')
        self.model = dict()

    def decay_esp(self):
        self.epsilon *= self.eps_decay_factor
        # self.epsilon = self.epsilon_final + self.eps_interval * exp(-1. * self.timestep / MAX_TIMESTEP)
        # self.epsilon = 0.3

    def action_arg_max(self, state):
        q_list = self.q_table[state[0], state[1]]
        action = np.random.choice(np.where(q_list == q_list.max())[0])
        return action

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.action_arg_max(state)

        return action

    def update_q(self, state, state_p, reward, action):
        old_q = self.q_table[state[0], state[1], action]
        action_p = self.action_arg_max(state_p)
        max_q_s_p = self.q_table[state_p[0], state_p[1], action_p]
        new_q = old_q + self.alpha*(reward + self.gamma*max_q_s_p - old_q)
        self.q_table[state[0], state[1], action] = new_q

    def get_new_reward(self, reward, *args, **kwargs):
        return reward

    def step(self, state, state_p, reward, action):
        self.decay_esp()
        self.update_q(state, state_p, reward, action)
        model_key = (state[0], state[1])

        if model_key not in self.model.keys():
            self.model[model_key] = dict()
            for a in self.action_space:
                if a != action:
                    self.model[tuple(state)][a] = [tuple(state), 0, 0]

        self.model[tuple(state)][action] = [tuple(state_p), reward, self.timestep]

        record_count = 0
        while record_count < self.n:
            record_count += 1
            sample_state = random.sample(list(self.model), 1)[0]
            if sample_state == model_key:
                continue
            sample_action = random.sample(list(self.model[sample_state]), 1)[0]
            sample_next_state, sample_reward, sample_time = self.model[sample_state][sample_action]
            sample_reward = self.get_new_reward(sample_reward, sample_state, sample_action, sample_time)
            self.update_q(sample_state, sample_next_state, sample_reward, sample_action)

        self.timestep += 1


class DynaQP(DynaQ):

    def get_new_reward(self, reward, state, action, last_time):
        bonus_reward = reward + self.k * sqrt(self.timestep - last_time)
        return bonus_reward


class CustomDynaQP(DynaQP):

    def act(self, state):

        q_list = self.q_table[state[0], state[1]]
        model_key = (state[0], state[1])
        last_visit = self.model.get(model_key, None)

        top = float("-inf")
        ties = []

        for aid in range(self.action_space_size):
            if not last_visit:
                bonus = 0
            else:
                last_time = last_visit.get(aid, [0])[-1]
                bonus = self.k * sqrt(self.timestep - last_time)
            new_q = q_list[aid] + bonus

            if new_q > top:
                top = new_q
                ties = []

            if new_q == top:
                ties.append(aid)

        return np.random.choice(ties)

# class DynaQQ


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    agent = DynaQ()
    eps_list = []
    for i in range(MAX_TIMESTEP):
        eps_list.append(agent.epsilon)
        agent.decay_esp()
        agent.timestep += 1

    x = np.arange(MAX_TIMESTEP)
    plt.figure()
    plt.plot(x, eps_list)
    plt.show()

