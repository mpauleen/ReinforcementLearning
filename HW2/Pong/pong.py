import gym
import gym.spaces
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
env = gym.make('Pong-v0')
obs = env.reset()
actions = ['RIGHT','LEFT']
n_actions = len(actions)
gamma = 0.99


## Baseline Network

class BaselineFunction():
    def __init__(self, learning_rate=0.001, state_size=6400, action_size=2,
                 output_size=1, hidden_state_size=16, name="BaselineFunction"):
        with tf.name_scope(name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, 80, 80, 2], name="inputs")
            self.reward_ = tf.placeholder(
                tf.float32, [None, ], name="expected_episode_rewards")
            
            self.conv1 = tf.layers.conv2d(self.inputs_, filters = 3, kernel_size = 5, 
                                     activation = tf.nn.relu, 
                                     kernel_initializer = tf.contrib.layers.xavier_initializer(), 
                                          name = 'features',
                                         reuse=True)
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)

            self.poolflat = tf.reshape(self.pool1, [-1, 38 * 38 * 3])

            self.fc1 = tf.contrib.layers.fully_connected(
                self.poolflat, hidden_state_size,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc2 = tf.contrib.layers.fully_connected(
                self.fc1, hidden_state_size,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc3 = tf.contrib.layers.fully_connected(
                self.fc2, 1,
                weights_initializer=tf.contrib.layers.xavier_initializer())
        with tf.name_scope("baseline_loss"):
            self.loss = tf.reduce_mean(tf.square(
                self.fc3 - self.reward_))
        with tf.name_scope("baseline_train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate, name = 'bopt')
            self.train = self.optimizer.minimize(self.loss, name = 'btrain')



## Policy Network

class PolicyGradient():
    def __init__(self, learning_rate=0.001, action_size=2,
                 hidden_state_size=16, name="PolicyGradient"):
        

        with tf.name_scope(name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, 80, 80, 2], name="inputs")
            self.actions_ = tf.placeholder(
                tf.int32, [None, action_size], name="actions")
            self.reward_ = tf.placeholder(
                tf.float32, [None, ], name="expected_episode_rewards")

            self.conv1 = tf.layers.conv2d(self.inputs_, filters = 3, kernel_size = 5, 
                                     activation = tf.nn.relu, 
                                     kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                         name = 'features')
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)

            self.poolflat = tf.reshape(self.pool1, [-1, 38 * 38 * 3])
            self.fc1 = tf.contrib.layers.fully_connected(
                self.poolflat, hidden_state_size,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc2 = tf.contrib.layers.fully_connected(
                self.fc1, hidden_state_size,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc3 = tf.contrib.layers.fully_connected(
                self.fc2, action_size,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None)
        with tf.name_scope("softmax"):
            self.softmax = tf.nn.softmax(self.fc3)
        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.actions_, logits=self.fc3)
            self.loss = tf.reduce_mean(
                self.cross_entropy * self.reward_, name = 'policy_loss')
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = self.optimizer.minimize(self.loss)


def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float).ravel(), [80,80])

def normalize_reward(r):
    discounted =[0]*len(r)
    cumulative = 0.0
    for i in reversed(range(len(r))):
        cumulative = cumulative * gamma + r[i]
        discounted[i] = cumulative
    mean = np.mean(discounted)
    std = np.std(discounted)
    discounted = (discounted - mean)/std
    return discounted

max_episodes = 10000

tf.reset_default_graph()

# initialize networks
policy_network = PolicyGradient(hidden_state_size=10)
baseline_network = BaselineFunction(hidden_state_size=10)

# set up pong environment
env = gym.make('Pong-v0')

# Initialize the simulation
env.reset()

# store epoch rewards
all_rewards = []

saver = tf.train.Saver()

# train Policy Gradient Network
with tf.Session() as sess: 
#    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./model.cpkt.meta')
    saver.restore(sess, 'model.cpkt')

    for i in range(max_episodes):
        
        r = []
        observations = []
        actions = []
        normalized_reward = []
        conv_outputs = []
        state = env.reset() # initialize the game by resetting. we assign the initial state to variable state. Every time the game finishes you need to reset the game.

        prev_state = preprocess(state)
        curr_state = preprocess(env.step(0)[0])
        episode_rewards = []

        while True: 

            # Stack current and previous frame
            stacked = np.concatenate((curr_state,prev_state)).reshape([-1,80,80,2])
            action_prob_dist = sess.run(policy_network.softmax, feed_dict={policy_network.inputs_:stacked})

            action = np.random.choice(range(n_actions), p=action_prob_dist.ravel()) #choose random action
            one_hot_action = np.zeros(n_actions)
            one_hot_action[action] = 1
            next_state, reward, done, info = env.step(2+action) #take the actions and see the next state, reward, if the game is finished or not and some info about the game.
            
            #keep track of rewards, states and actions
            r.append(reward)
            observations.append(stacked)
            actions.append(one_hot_action)
            prev_state = curr_state
            curr_state = preprocess(next_state)

            if done:
                normalized_reward = normalize_reward(r)

                # get baseline adjustment
                baseline_ = sess.run(baseline_network.fc3, feed_dict={baseline_network.inputs_ : np.vstack(observations) })
                exp_rewards_b = normalized_reward - np.hstack(baseline_)
                
                # train baseline network
                _, _= sess.run([baseline_network.loss, baseline_network.train], 
                            feed_dict={baseline_network.inputs_: np.vstack(observations),
                            baseline_network.reward_: normalized_reward })

                loss_, _ = sess.run([policy_network.loss, policy_network.train], 
                                    feed_dict={policy_network.inputs_: np.vstack(observations),
                                                                 policy_network.actions_: actions,
                                                                 policy_network.reward_: exp_rewards_b
                                                                })
                break

        all_rewards.append(np.sum(r))
        if i % 100 == 0:
            print("Epoch: %s    Length: %s    Reward: %s    L50 Reward: %s" %(i, len(r), all_rewards[i], np.mean(all_rewards[i-50:i])))
            with open('result.txt','a') as f:
                f.write("\n Epoch: %s    Length: %s    Reward: %s    L50 Reward: %s" %(i, len(r), all_rewards[i], np.mean(all_rewards[i-50:i])))
            saver.save(sess, './model.cpkt')
