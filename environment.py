import numpy as np
from matplotlib import pyplot as plt

from settings import *


# generate same grid world map in the textbook
def gen_maze():
    m1 = np.zeros((MAZE_H, MAZE_W))
    m2 = np.zeros((MAZE_H, MAZE_W))

    m1[0, 8] = GOAL
    m1[5, 3] = START
    m1[3, 0:8] = WALL

    m2[0, 8] = GOAL
    m2[5, 3] = START
    m2[3, 1:9] = WALL
    return m1, m2


class Maze:
    def __init__(self):
        self.stepcount = 0
        self.w = MAZE_W
        self.h = MAZE_H

        self.maze_1, self.maze_2 = gen_maze()
        self.cur_maze = self.maze_1
        self.cur_start = self.find_value(START)[0]

        self.actions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype='int')

    def reset(self):
        return np.array(self.cur_start)

    def find_value(self, value):
        pos = np.where(self.cur_maze == value)
        pos = np.array([[pos[0][x], pos[1][x]] for x in range(len(pos[0]))])
        return pos

    def step(self, state, action):
        if self.stepcount == SWITCH_TIMESTEP-1:
            self.cur_maze = self.maze_2
            self.cur_start = self.find_value(START)[0]

        self.stepcount += 1
        reward = REWARD_NORMAL
        terminal = False
        movement = self.actions[action]
        new_state = state + movement
        new_row, new_col = new_state

        if new_row < 0 or new_row > self.h - 1 or new_col < 0 or new_col > self.w - 1:
            new_state = state
            reward = REWARD_BLOCK
        elif self.cur_maze[new_row, new_col] == WALL:
            new_state = state
            reward = REWARD_BLOCK
        elif self.cur_maze[new_row, new_col] == GOAL:
            reward = REWARD_GOAL
            terminal = True
        return reward, new_state, terminal


def draw_maze(m):
    cmap = plt.cm.binary
    bounds = [0, 1, 2, 3]
    norm = plt.Normalize(bounds[0], bounds[-1])
    colors = cmap(norm(m))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the maze
    ax.imshow(colors, interpolation='none')

    # Loop through the array and add the starting and finishing points
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] == 1:
                ax.text(j, i, 's', ha='center', va='center', color='red', fontsize=20)
            elif m[i, j] == 2:
                ax.text(j, i, 'g', ha='center', va='center', color='green', fontsize=20)

            # Draw boundaries around each box in the maze
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    maze = Maze()
    draw_maze(maze.maze_1)
    draw_maze(maze.maze_2)
    pos_3 = maze.find_value(3)
    print(pos_3)
