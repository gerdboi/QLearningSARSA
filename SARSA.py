from collections import deque
import gym
import random
import numpy as np
import time
import pickle

from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":




    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)

    #Once again, inspiration was drawn from https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e
    #I was much closer to solving this function on my own before I hit a wall, and this site helped me put the final pieces in place. I wish I could have come up with the solutions
    #To both algorithms on my own, but in my plugging and with the help of this site and more than a few YouTube videos, I feel very confident in my grasp of this subject.
    Q_table = np.zeros((env.observation_space.n, env.action_space.n)) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        episode_reward = 0
        state = env.reset()

        #Find the first action before entering loop
        randN = np.random.uniform(0,1)
        if randN > EPSILON:
            action = np.argmax(Q_table[state,:])
        else:
            action = env.action_space.sample()

        
        for k in range(100):

            #Immediately take the action and get the next state.
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            #Find the second action
            randN = np.random.uniform(0,1)
            if randN > EPSILON:
                next_action = np.argmax(Q_table[next_state,:])
            else:
                next_action = env.action_space.sample()

            #Update the Q Table
            if done:
                Q_table[state,action] += (LEARNING_RATE) * (reward - Q_table[state,action])
            else:
                target = reward + DISCOUNT_FACTOR*Q_table[next_state,next_action]
                Q_table[state,action] += (LEARNING_RATE)*(target - Q_table[state,action])   

            #Assign the next starting state and action
            state = next_state
            action = next_action
            
            #Break if done.
            if done:
                break
        episode_reward_record.append(episode_reward)

        EPSILON = EPSILON*EPSILON_DECAY
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################



