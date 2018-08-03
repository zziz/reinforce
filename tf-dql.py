import random, gym
import numpy as np
from collections import deque
import tensorflow as tf

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.state = tf.placeholder(shape=[1, self.state_size], dtype=tf.float32)
        self.reward = tf.placeholder(shape=[1, self.action_size], dtype=tf.float32)

        self.model = self._model(self.state)
        self.saver = tf.train.Saver()
        self.cost = tf.reduce_mean(tf.pow(self.model - self.reward, 2) / 2)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def _model(self, state):
        nn = tf.layers.dense(state, 24, activation=tf.nn.relu)
        nn = tf.layers.dense(nn, 24, activation=tf.nn.relu)
        nn = tf.layers.dense(nn, self.action_size)
        return nn

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.sess.run([self.model], feed_dict={self.state: state})
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.sess.run([self.model], feed_dict={self.state: next_state}))
            target_f = self.sess.run([self.model], feed_dict={self.state: state})
            target_f[0][0][action] = target
            _, cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.state: state, self.reward: target_f[0]})
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.saver.restore(self.sess, name)

    def save(self, name):
        self.saver.save(self.sess, name)

    def play(self, env, n_games):
        for game in range(n_games):
            state = env.reset()
            state = np.reshape(state, [1, env.observation_space.shape[0]])
            total_reward = 0
            while True:
                env.render()
                action =  np.argmax(self.sess.run([self.model], feed_dict={self.state: state}))
                state, reward, done, _ = env.step(action)
                state = np.reshape(state, [1, env.observation_space.shape[0]])
                total_reward += reward
                if done: break
            print("%d Reward: %s" % (game, total_reward))

if __name__ == "__main__":
    n_episodes = 1000
    batch_size = 32
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/model.ckpt")

    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.4}".format(e, n_episodes, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("./save/model.ckpt")
    agent.play(env, 100)
