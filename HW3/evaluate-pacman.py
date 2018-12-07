import matplotlib as mpl
mpl.use('Agg')

import gym
import gym.spaces
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from helper import Replay, History, DQN, param_copier

log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt, filename = 'eval-pacman.log')
logger = logging.getLogger(__name__)


N_GAMES = 1000

QNetwork = DQN(name='QNetwork', action_size = 9, y_shape = 88)
saver = tf.train.Saver()
history = History()
rewards = []


mspacman_color = 210 + 164 + 74

def preprocess(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80)

env = gym.make('MsPacman-v0')
env.reset()

with tf.Session() as sess: 
    
    saver.restore(sess, "pacman-model/2/model7160.ckpt")
    
    
    for g in range(N_GAMES):
        state = env.reset()
        prepro = preprocess(state)

        # Initialize history
        for i in range(5):
            history.add(prepro)
            
        # Initialize total reward
        total_reward = 0
        
        while True:
            old_hist = history.as_numpy()

            action = QNetwork.predict(sess, [old_hist])
            new_state, reward, done, _ = env.step(np.argmax(action))
            
            new_prepro = preprocess(new_state)
            history.add(new_prepro)
            new_hist = history.as_numpy()
            if done:
                rewards.append(total_reward)
                logger.info('Game: {}\tReward: {}'.format(g, total_reward))
                print('Game: {}\tReward: {}'.format(g, total_reward))
                break
            else: 
                # Update reward total
                total_reward += reward
print(rewards)
plt.hist(rewards)
plt.xlabel('Game Reward')
plt.ylabel('Frequency')
plt.title('Pacman Reward Frequency')
plt.savefig('pacman-results.png')



        
