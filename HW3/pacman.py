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


log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt, filename = 'pacman.log')
logger = logging.getLogger(__name__)


N_EPOCHS = 100000
BATCH_SIZE = 64
DISCOUNT_RATE = 0.99
TARGET_UPDATE = 1000

mspacman_color = 210 + 164 + 74

def preprocess(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80)

env = gym.make('MsPacman-v0')
obs = env.reset()


log_root = './pacman_logs/'


run_num = str(len(os.listdir(log_root)))
os.mkdir(os.path.join(log_root, run_num))
os.mkdir(os.path.join('pacman-model', run_num))
logdir = os.path.join(log_root, run_num)
logdir = os.path.join(log_root, run_num)


tf.reset_default_graph()

QNetwork = DQN(name='QNetwork', action_size = 9, y_shape = 88)
target = DQN(name='Target', action_size = 9, y_shape = 88)

saver = tf.train.Saver()

buffer = Replay()
history = History()

with tf.Session() as sess: 
    writer = tf.summary.FileWriter(logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    count = 0

    # Initialize History
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

        one_hot_action = np.zeros(9)
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
            new_hist = history.as_numpy()

            one_hot_action = np.zeros(9)
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

            if done:
                # Reset history
                state = env.reset()
                prepro = preprocess(state)
                for i in range(5):
                    history.add(prepro)

                avg_max_Q = np.mean(result)
                loss, summary = QNetwork.summarize(sess, loss, avg_max_Q, reward_total)

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
                saver.save(sess, "./pacman-model/{0}/model{1}.ckpt".format(run_num, epoch))

