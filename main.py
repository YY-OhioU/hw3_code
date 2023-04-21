import random

import numpy as np
from matplotlib import pyplot as plt
from environment import Maze
from agents import DynaQ, DynaQP, CustomDynaQP

from settings import *


def run_tests(agent_cls, *args, **kwargs):
    np.random.seed(101)
    random.seed(101)

    maze = Maze()
    action_space = tuple([x for x in range(maze.actions.shape[0])])
    reward_hist = np.zeros((NUM_TEST, MAX_TIMESTEP))
    for epoch in range(NUM_TEST):
        hit_count = 0
        maze = Maze()
        agent = agent_cls(*args, **kwargs, action_space=action_space,)
        state = maze.reset()
        terminate = False
        accumulated_reward = 0
        for step in range(MAX_TIMESTEP):
            action = agent.act(state)
            reward, new_state, terminate = maze.step(state, action)
            if step == SWITCH_TIMESTEP -1:
                print(f"[Epoch {epoch}] before-switch {hit_count}")
            agent.step(state, new_state, reward, action)
            state = new_state
            accumulated_reward += reward
            reward_hist[epoch, step] = accumulated_reward

            if terminate:
                state = maze.reset()
                hit_count += 1
                # print(f"[{epoch}_{step}]find")
        print(f"[Epoch {epoch}]: {hit_count}")
    return reward_hist


if __name__ == '__main__':
    X = [x for x in range(MAX_TIMESTEP)]

    print("Dyna-Q =======================")
    dyna_q_reward_hist = run_tests(DynaQ)
    dyna_q_reward_average = np.average(dyna_q_reward_hist, axis=0)
    fig, ax = plt.subplots()
    plt.plot(X, dyna_q_reward_average, label='Dyna-Q', color='red')

    print("Dyna-Q+ =======================")
    dyna_qp_reward_hist = run_tests(DynaQP)
    dyna_qp_reward_average = np.average(dyna_qp_reward_hist, axis=0)
    plt.plot(X, dyna_qp_reward_average, label='Dyna-Q+', color='blue')

    print("Custom_Dyna-Q+ =======================")
    dyna_cqp_reward_hist = run_tests(CustomDynaQP)
    dyna_cqp_reward_average = np.average(dyna_cqp_reward_hist, axis=0)
    plt.plot(X, dyna_cqp_reward_average, label='custom_Dyna-Q+', color='m')

    plt.axvline(x=SWITCH_TIMESTEP-1, linestyle='--', color='grey', alpha=0.4)

    plt.legend()
    plt.show()


