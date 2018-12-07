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
logging.basicConfig(level=logging.INFO, format=log_fmt, filename = 'eval-breakout.log')
logger = logging.getLogger(__name__)

N_GAMES = 1000

QNetwork = DQN(name='QNetwork', action_size = 4, y_shape = 80)
saver = tf.train.Saver()
history = History()
rewards = []


def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image = image/255.0
    return np.reshape(image.astype(np.float).ravel(), [80,80])

env = gym.make('Breakout-v0')
env.reset()

with tf.Session() as sess: 
    
    saver.restore(sess, "breakout-model/3/model800.ckpt")
    
    
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
                
logger.info('Average Reward: {}'.format(np.mean(rewards)))

logger.info('Max Reward: {}'.format(max(rewards)))

print(rewards)
plt.hist(rewards)
plt.xlabel('Game Reward')
plt.ylabel('Frequency')
plt.title('Breakout Reward Frequency')
plt.savefig('breakout-results.png')



        
