import gym
import random
import numpy as np
import time
from collections import deque
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

    episode_reward_record = deque(maxlen=100)
    
    #I took inspiration from https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187, I am new to Python, and Dictionaries are not
    #a strong suit of mine, so I was able to make more sense of the concept as a table.
    n_observations = env.observation_space.n
    n_actions = env.action_space.n
    Q_table = np.zeros((n_observations,n_actions))

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        state = env.reset()

        for k in range(100):
        
            best_action = np.argmax(Q_table[state,:])
            
            randN = np.random.uniform(0,1)
            if randN > EPSILON:
                action = best_action
                
            else:
                action = env.action_space.sample()
            
            next_state, reward, done, info = env.step(action)
            best_next_action = np.argmax(Q_table[next_state])
            Q_table[state,action] = (1 - LEARNING_RATE)*Q_table[state,action] + LEARNING_RATE*(reward + DISCOUNT_FACTOR*Q_table[next_state,best_next_action])
            episode_reward += reward
            
            if done:
                break
            state = next_state

        episode_reward_record.append(episode_reward)

            
        EPSILON = EPSILON * EPSILON_DECAY
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################







