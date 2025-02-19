from env import Maze
from q_learning import QLearning

def update():
    step_all=0
    avg_step=[]
    sucess_rate=[]
    alldone=0
    cnt=0
    for episode in range(100):
        state = env.reset()
        step_count = 0
        while True:
            env.render()
            action = RL.choose_action(str(state))
            state_, reward, done = env.step(action)

            step_count += 1
            RL.learn(str(state), action, reward, str(state_))
            state = state_

            if done==0 or done==1:
                print("Round {} end. Total step: {}\n".format(episode+1, step_count))
                s="Round {} end. Total step: {}\n".format(episode+1, step_count)
                env.message(s,done)
                step_all += step_count
                if done==1:
                  alldone+=done
                if (episode+1)%25==0:
                    avg_step.append(step_all/(25))
                    sucess_rate.append(alldone/(25)*100)
                    cnt+=1
                    step_all = 0
                    alldone = 0
                break

    print('Game end')

    print(f"{'Game Count':<10} | {'Success rate of the last 25 times(%)':<40} | {'Average number of steps in the last 25 times':<40}")
    print("-" * 100)
    for i in range(0,cnt):
        print(f"{25*(i+1):<13} | {sucess_rate[i]:<40.2f} | {avg_step[i]:<40.2f}")

    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearning(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

    print('\nQ Table:')
    print(RL.q_table)