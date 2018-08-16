import gym
import numpy as np
from RL_brain import DeepQNetwork
import time

def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()
        rSum = 0
        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward

            observation_, reward, done,_ = env.step(action)
            rSum+=reward
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 10 == 0):
                print("learn"+str(step))
                RL.learn()


            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print("回合结束:reword:%s"%rSum)
                break
            step += 1
            time.sleep(0.01)

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = gym.make('Pooyan-v0')

    RL = DeepQNetwork(6, 600,
                      learning_rate=0.0001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )

    for i in range(1,100):
        run_maze()
