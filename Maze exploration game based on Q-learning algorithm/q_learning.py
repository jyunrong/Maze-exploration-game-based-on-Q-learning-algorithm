"""
Q Learning 算法
"""

import numpy as np
import pandas as pd


class QLearning:
    def __init__(self, actions, learning_rate=0.01, discount_factor=0.9, e_greedy=0.1):
        self.actions = actions        # action 列表
        self.lr = learning_rate       # 学习速率
        self.gamma = discount_factor  # 折扣因子
        self.epsilon = e_greedy       # 贪婪度
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)  # Q 表

    # 如果还没有当前 state, 就插入一组全 0 数据, 作为这个 state 的所有 action 的初始值

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = pd.concat(
                [
                    self.q_table,
                    pd.DataFrame(
                        [[0] * len(self.actions)],
                        columns=self.q_table.columns,
                        index=[state]
                    )
                ]
            )

    # 根据 state 来选择 action

    def choose_action(self, state):
        self.check_state_exist(state)
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[state, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

