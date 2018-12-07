import numpy as np
import random
import tensorflow as tf

from collections import deque


class Replay():
    def __init__(self, buff_size = 100000, n = 64):
        self.buffer = deque(maxlen = buff_size)
        self.n = n
    def add(self, ep):
        self.buffer.append(ep)
        
    def sample(self):
        return list(random.sample(self.buffer, self.n)).copy()
    
    
class History():
    def __init__(self, history_size = 4):
        self.buffer = deque(maxlen = history_size)
    
    def add(self, frame):
        self.buffer.append(frame)
    
    def as_numpy(self):
        buff_list = list(self.buffer)
        return np.stack(buff_list, axis = -1).copy()
    
    
class DQN():
    def __init__(self, lr = 0.0001,
                 action_size = 4,
                 history_size = 4,
                 y_shape = 80,
                 name = 'Network'
                ):
    
        self.action_size = action_size
        self.epsilon_decay = 0.01
        self.epsilon_step = 0
        self.epsilon_min = 0.01
        
        with tf.variable_scope(name):
            self.scope = name

            self.inputs_ = tf.placeholder(tf.float32, [None, y_shape, 80, history_size], name='inputs')
            self.target_preds_ = tf.placeholder(tf.float32, [None,], name="expected_future_rewards")
            self.Q = tf.placeholder(tf.float32, [None,], name="chosen_action_pred")
            self.actions_ = tf.placeholder(tf.float32, shape=[None, action_size], name='actions')
            self.avg_max_Q_ = tf.placeholder(tf.float32, name="avg_max_Q")
            self.reward_ = tf.placeholder(tf.float32, name="reward")
            self.epoch_loss_ = tf.placeholder(tf.float32, name="epoch_loss")
            # Define Network Topology
            
            # Three Conv Layers
            
            self.conv1 = tf.layers.conv2d(
                inputs = self.inputs_, 
                filters = 16,
                kernel_size = [8,8],
                strides = [4,4],
                padding = "VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.relu)

            self.conv2 = tf.layers.conv2d(
                inputs = self.conv1, 
                filters = 8,
                kernel_size = [4,4],
                strides = [2,2],
                padding = "VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.relu)
        
            self.flatten = tf.layers.flatten(self.conv2)
            
            self.fc1 = tf.layers.dense(
                self.flatten, 512, activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")
            self.predictions = tf.layers.dense(
                self.fc1, units=action_size,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                activation=tf.nn.relu)

        with tf.variable_scope("Q"):
            self.Q = tf.reduce_sum(tf.multiply(self.predictions, self.actions_), axis=1)
        
        with tf.variable_scope("loss"):
            self.h_loss = tf.losses.huber_loss(self.target_preds_, self.Q)
            self.loss = tf.reduce_mean(self.h_loss)

        with tf.variable_scope("train"):
            self.optimizer = tf.train.RMSPropOptimizer(lr, momentum=0.95, epsilon=0.01)
            self.train = self.optimizer.minimize(self.loss)

        with tf.variable_scope("summaries"):
            tf.summary.scalar("loss", self.epoch_loss_)
            tf.summary.scalar("avg_max_Q", self.avg_max_Q_)
            tf.summary.scalar("reward", self.reward_)
            self.summary_op = tf.summary.merge_all()
            
    def predict(self, sess, state):
        result = sess.run(self.predictions, feed_dict={self.inputs_: state})
        return result
    
    def update(self, sess, state, action, target_preds):
        feed_dict = {self.inputs_: state, 
                    self.actions_: action, 
                    self.target_preds_: target_preds}
        loss = sess.run([self.loss, self.train], feed_dict=feed_dict)
        return loss

    def predict_explore(self, sess, state):
        
        epsilon = self.epsilon_min + (1 - self.epsilon_min) * np.exp(-self.epsilon_decay * self.epsilon_step)
        pick = np.random.rand() # Uniform random number generator
        
        if pick < epsilon: # If off policy -- random action
            action = np.random.randint(0,self.action_size)
        else: # If on policy
            action = np.argmax(self.predict(sess, [state]))
        
        self.epsilon_step += 1
        return action
    
    def summarize(self, sess, loss, avg_max_Q, reward):
        summary = sess.run(self.summary_op, feed_dict={self.epoch_loss_: loss,
                                                            self.avg_max_Q_: avg_max_Q, 
                                                        self.reward_: reward})
        return loss, summary
    
def param_copier(sess, q_network, target_network):
    
    # Get and sort parameters
    q_params = [t for t in tf.trainable_variables() if t.name.startswith(q_network.scope)]
    q_params = sorted(q_params, key=lambda v: v.name)
    t_params = [t for t in tf.trainable_variables() if t.name.startswith(target_network.scope)]
    t_params = sorted(t_params, key=lambda v: v.name)
    
    # Assign Q-Parameters to Target Network
    updates = []
    for q_v, t_v in zip(q_params, t_params):
        update = t_v.assign(q_v)
        updates.append(update)
    
    sess.run(updates)