import gym
import gym.spaces
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf

from collections import deque

from helper import Replay, History, DQN, param_copier

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt, filename = 'breakout.log')
logger = logging.getLogger(__name__)


N_EPOCHS = 100000
BATCH_SIZE = 64
DISCOUNT_RATE = 0.95
TARGET_UPDATE = 1000


def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image = image/255.0
    return np.reshape(image.astype(np.float).ravel(), [80,80])


env = gym.make('Breakout-v0')
obs = env.reset()

log_root = './breakout_logs/'

run_num = str(len(os.listdir(log_root)))
os.mkdir(os.path.join(log_root, run_num))
os.mkdir(os.path.join('breakout-model', run_num))
logdir = os.path.join(log_root, run_num)

tf.reset_default_graph()

QNetwork = DQN(name='QNetwork')
target = DQN(name='Target')

saver = tf.train.Saver()

buffer = Replay(buff_size = 200000)
history = History()

with tf.Session() as sess: 
    writer = tf.summary.FileWriter(logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    # Set up count for network reset
    count = 0

    # Set up history for episode
    state = env.reset()
    prepro = preprocess(state)
    for i in range(5):
        history.add(prepro)

    # Fill The Buffer
    for i in range(10000):
        old_hist = history.as_numpy()
        action = QNetwork.predict_explore(sess, old_hist)
        new_state, reward, done, _ = env.step(action)

        new_prepro = preprocess(new_state)
        history.add(new_prepro)
        new_hist = history.as_numpy()

        one_hot_action = np.zeros(4)
        one_hot_action[action] = 1

        # Add step to buffer
        buffer.add([old_hist, one_hot_action, new_hist, reward, done])

        # If done, reset history
        if done: 
            state = env.reset()
            prepro = preprocess(state)
            for i in range(5):
                history.add(prepro)

    # Train
    for epoch in range(N_EPOCHS):
        # For Tensorboard
        result = [] 
        reward_total = 0

        # Set Up Memory
        state = env.reset()
        prepro = preprocess(state)
        for i in range(5):
            history.add(prepro)

        while True: 
            # Add M to buffer (following policy)
            old_hist = history.as_numpy()

            action = QNetwork.predict_explore(sess, old_hist)
            new_state, reward, done, _ = env.step(action)

            new_prepro = preprocess(new_state)
            history.add(new_prepro)
            hew_hist = history.as_numpy()

            one_hot_action = np.zeros(4)
            one_hot_action[action] = 1

            # Add step to buffer
            buffer.add([old_hist, one_hot_action, new_hist, reward, done])

            sample = np.array(buffer.sample())
            state_b, action_b, new_state_b, reward_b, done_b = zip(*sample)
            # Find max Q-Value per batch for progress
            Q_preds = sess.run(QNetwork.Q, 
                                feed_dict={QNetwork.inputs_: state_b,
                                QNetwork.actions_: action_b})
            result.append(np.max(Q_preds))

            # Q-Network
            T_preds = []
            TPreds_batch = target.predict(sess, new_state_b)
            for i in range(BATCH_SIZE):
                terminal = done_b[i]
                if terminal:
                    T_preds.append(reward_b[i])
                else:
                    T_preds.append(reward_b[i] + DISCOUNT_RATE * np.max(TPreds_batch[i]))

            # Update Q-Network
            loss, _ = QNetwork.update(sess, state_b, action_b, T_preds)

            # If simulation done, stop
            if done:
                # Reset history
                state = env.reset()
                prepro = preprocess(state)
                for i in range(5):
                    history.add(prepro)
                # Tensorboard
                avg_max_Q = np.mean(result)
                loss, summary = QNetwork.summarize(sess, loss, avg_max_Q, reward_total)
                # Log and save models
                logger.info("Epoch: {0}\tAvg Reward: {1}".format(epoch, avg_max_Q))
                writer.add_summary(summary, epoch)
                break
            else: 
                reward_total = reward_total + reward

            # Save target network parameters every epoch
            count += 1
            if count % TARGET_UPDATE == 0:
                param_copier(sess, QNetwork, target)

        # save model
        if epoch % 20 == 0:
                saver.save(sess, "./breakout-model/{0}/model{1}.ckpt".format(run_num, epoch))

