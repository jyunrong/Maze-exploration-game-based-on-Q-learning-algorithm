"""
Q Learning 例子的 Maze（迷宫） 环境
黄色圆形 :   机器人
红色方形 :   炸弹     [reward = -1]
绿色方形 :   宝藏     [reward = +1]
其他方格 :   平地     [reward = 0]
"""

import sys
import time
import numpy as np
import tkinter as tk

WIDTH = 6   # 迷宫的宽度
HEIGHT = 6  # 迷宫的高度
UNIT = 40   # 每个方块的大小（像素值）


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # 上，下，左，右 四个 action（动作）
        self.n_actions = len(self.action_space)   # action 的数目
        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))  # Tkinter 的几何形状
        self.build_maze()

    def build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                width=WIDTH * UNIT,
                                height=HEIGHT * UNIT)

        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])

        robot_center = origin + np.array([0, UNIT * 5])
        self.robot = self.canvas.create_oval(
            robot_center[0] - 15, robot_center[1] - 15,
            robot_center[0] + 15, robot_center[1] + 15,
            fill='yellow')

        bomb1_center = origin + np.array([UNIT*5, UNIT*2])
        self.bomb1 = self.canvas.create_rectangle(
            bomb1_center[0] - 15, bomb1_center[1] - 15,
            bomb1_center[0] + 15, bomb1_center[1] + 15,
            fill='red')

        bomb2_center = origin + np.array([UNIT*3, UNIT*4])
        self.bomb2 = self.canvas.create_rectangle(
            bomb2_center[0] - 15, bomb2_center[1] - 15,
            bomb2_center[0] + 15, bomb2_center[1] + 15,
            fill='red')

        bomb3_center = origin + np.array([UNIT, UNIT])
        self.bomb3 = self.canvas.create_rectangle(
            bomb3_center[0] - 15, bomb3_center[1] - 15,
            bomb3_center[0] + 15, bomb3_center[1] + 15,
            fill='red')

        treasure_center = origin + np.array([UNIT * 5, 0])
        self.treasure = self.canvas.create_rectangle(
            treasure_center[0] - 15, treasure_center[1] - 15,
            treasure_center[0] + 15, treasure_center[1] + 15,
            fill='green')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.robot)
        origin = np.array([20, 20])
        robot_center = origin + np.array([0, UNIT * 5])
        self.robot = self.canvas.create_oval(
            robot_center[0] - 15, robot_center[1] - 15,
            robot_center[0] + 15, robot_center[1] + 15,
            fill='yellow')
        return self._discretize_state(self.canvas.coords(self.robot))

    def show_message(self, message):
        background = self.canvas.create_rectangle(
            WIDTH * UNIT / 2 - 100, HEIGHT * UNIT / 2 - 20,
            WIDTH * UNIT / 2 + 100, HEIGHT * UNIT / 2 + 20,
            fill="white", outline="grey"
        )
        self.message_box = self.canvas.create_text(
            WIDTH * UNIT / 2, HEIGHT * UNIT / 2,
            text=message, fill="black", font=('Arial', 10)
        )
        self.after(200, lambda: self.remove_message(background))

    def remove_message(self, background):
        self.canvas.delete(self.message_box)
        self.canvas.delete(background)

    def step(self, action):
        s = self.canvas.coords(self.robot)
        base_action = np.array([0, 0])
        if action == 0:  # 上
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 下
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 右
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # 左
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # 移动机器人
        self.canvas.move(self.robot, base_action[0], base_action[1])

        # 下一个 state
        s_ = self.canvas.coords(self.robot)

        # 奖励机制
        if s_ == self.canvas.coords(self.treasure):
            reward = 1
            done = 1
            s_ = 'terminal'
            print("WIN!")
        elif s_ == self.canvas.coords(self.bomb1) or s_ == self.canvas.coords(self.bomb2) or s_ == self.canvas.coords(
                self.bomb3):
            reward = -1
            done = 0
            s_ = 'terminal'
            print("Bomb! GAME OVER!")
        else:
            reward = 0
            done = 2

        # 返回离散化的网格编号
        return self._discretize_state(s_), reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

    def _discretize_state(self, coords):
        if coords == 'terminal':
            return 'terminal'
        x1 = coords[2]
        y1 = coords[3]
        grid_x = int((x1 + 5) // UNIT)
        grid_y = int((y1 + 5) // UNIT)
        return f"({grid_x}, {grid_y})"

    def message(self,s,done):
        if done==1:
            self.show_message("\nWIN!\n{}".format(s))
        else:
            self.show_message("\nBomb! GAME OVER!\n{}".format(s))
