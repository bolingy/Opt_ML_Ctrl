#!/usr/bin/env python

#import pyglet
import gym
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import numpy as np
import random

import matplotlib.pyplot as plt

try:
    xrange = xrange
except:
    xrange = range

env = gym.make('CartPole-v0')

gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def reset_gradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

class agent():
    def __init__(self, lr, s_size, a_size, h_size, scope_name):
        # These lines established the feed-forward part of the network.
        #  The agent takes a state and produces an action.
        self.scope = scope_name
        with tf.name_scope(self.scope):
            self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            # weight for layers
            with tf.variable_scope(self.scope):
                self.W0 = tf.get_variable('W1', shape=[s_size, h_size])
                self.W1 = tf.get_variable('W2', shape=[h_size, h_size])
                self.W2 = tf.get_variable('W3', shape=[h_size, a_size])

            # layers
            self.layer0 = tf.nn.relu(tf.matmul(self.state_in, self.W0))
            self.layer1 = tf.nn.relu(tf.matmul(self.layer0, self.W1))
            self.output = tf.nn.softmax(tf.matmul(self.layer1, self.W2))

            self.chosen_action = tf.argmax(self.output, 1)

            # The next six lines establish the training proceedure.
            # We feed the reward and chosen action into the network
            # to compute the loss, and use it to update the network.
            self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

            self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1]\
                           + self.action_holder
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

            self.tvars = tf.trainable_variables(scope=self.scope)
            self.gradient_holders = []
            for idx, var in enumerate(self.tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)

            self.gradients = tf.gradients(self.loss, self.tvars)

            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))

    def save_model(self, save_path, tf_sess):
        print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,\
                                                          scope=self.scope)
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,\
                                                          scope=self.scope))
        saver.save(tf_sess, save_path)

    def restore_model(self, load_path, tf_sess):
        # reset default graph may needed
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, \
                                                          scope=self.scope))
        saver.restore(tf_sess, load_path)



'''
class adversarial_trainer():

    def __init__(self, protagonist_agent, adversary_agent):
        self.protagonist_agent = protagonist_agent
        self.adversary_agent = adversary_agent

    def train_protagonist(self, episodes):
'''

def train_agent(sess, protagonistAgent, adversaryAgent, gradBuffer_p, gradBuffer_a, p_a_switch, total_episodes = 500, max_ep = 999, update_frequency = 5):
    i = 0
    total_reward = []
    total_lenght = []

    if p_a_switch == 1:
        update_agent = protagonistAgent
        gradBuffer = gradBuffer_p
        r_ops = 1.0
    else:
        update_agent = adversaryAgent
        gradBuffer = gradBuffer_a
        r_ops = -1.0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0.0
        ep_history = []

        # update process
        for j in range(max_ep):
            # Probabilistically pick an action given our network outputs.
            # get ady action
            a_dist = sess.run(adversaryAgent.output, feed_dict={adversaryAgent.state_in: [s]})
            a_a = np.random.choice(a_dist[0], p=a_dist[0])
            a_a = np.argmax(a_dist == a_a)
            if a_a == 1:
                s[2] += 0.005
            else:
                s[2] -= 0.005

            # get protg action
            p_dist = sess.run(protagonistAgent.output, feed_dict={protagonistAgent.state_in: [s]})
            p_a = np.random.choice(p_dist[0], p=p_dist[0])
            p_a = np.argmax(p_dist == p_a)
            a = p_a

            s1, r, d, _ = env.step(a)  # Get our reward for taking an action given a bandit.
            r = r*r_ops
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r

            if d == True:
                # Update the network.
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {update_agent.reward_holder: ep_history[:, 2],
                             update_agent.action_holder: ep_history[:, 1],
                             update_agent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(update_agent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(update_agent.gradient_holders, gradBuffer))
                    _ = sess.run(update_agent.update_batch, feed_dict=feed_dict)
                    reset_gradBuffer(gradBuffer)

                total_reward.append(running_reward)
                total_lenght.append(j)
                break

                # Update our running tally of scores.
        if i % 10 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1



'''
    Train the policy Agent
'''

tf.reset_default_graph()  # Clear the Tensorflow graph.

# create two agent
protagonistAgent = agent(lr=1e-2, s_size=4, a_size=2, h_size=8, scope_name='protagonist')
adversaryAgent = agent(lr=1e-2, s_size=4, a_size=2, h_size=8, scope_name='adversary')

total_episodes = 2000  #5000 Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    gradBuffer_p = sess.run(protagonistAgent.tvars)
    gradBuffer_a = sess.run(adversaryAgent.tvars)
    reset_gradBuffer(gradBuffer_p)
    reset_gradBuffer(gradBuffer_a)

    render_flag = False

    for idx in range(1):
        # train protg
        train_agent(sess, protagonistAgent, adversaryAgent, gradBuffer_p, gradBuffer_a, 1)
        # train ady
        train_agent(sess, protagonistAgent, adversaryAgent, gradBuffer_p, gradBuffer_a, 0)



    # try save
    path = '/home/bolingy/Desktop/Opt_ML_Ctrl/models/ad0_model.ckpt'
    adversaryAgent.save_model(path, sess)


'''
if __name__ == '__main__':
    # remove everything from tf default graph
    tf.reset_default_graph()
    myAgent = agent(lr=1e-2, s_size=4, a_size=2, h_size=8, scope_name='policy')  # Load the agent.
    adAgent = agent(lr=1e-2, s_size=4, a_size=2, h_size=8, scope_name='adversary')
    s = env.reset()
    with tf.Session() as sess:
        #print('w0: %s' % myAgent.W0.eval())
        path = '/home/bolingy/Desktop/Opt_ML_Ctrl/models/pi0_model.ckpt'
        myAgent.restore_model(path, sess)
        adpath = '/home/bolingy/Desktop/Opt_ML_Ctrl/models/ad0_model.ckpt'
        adAgent.restore_model(adpath, sess)

        total_epi = 1000
        total_reward = []
        for i in range(total_epi):
            a_dist = sess.run(adAgent.output, feed_dict={adAgent.state_in: [s]})
            a_a = np.random.choice(a_dist[0], p=a_dist[0])
            a_a = np.argmax(a_dist == a_a)
            if a_a == 1:
                s[2] += 0.05
            else:
                s[2] -= 0.05

            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)
            s1, r, d, _ = env.step(a)
            total_reward.append(r)
            env.render()
            s = s1
            if i % 100 == 0:
                print(np.mean(total_reward[-100:]))
            i += 1
'''