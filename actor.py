import tensorflow as tf
import numpy as np

class ActorNetwork(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        # 直接在第一个 Dense 层指定 input_shape，而不使用 InputLayer
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(3 * embedding_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='tanh')
        ])
        
    def call(self, x):
        return self.fc(x)

class Actor(object):
    
    def __init__(self, embedding_dim, hidden_dim, learning_rate, state_size, tau):
        
        self.embedding_dim = embedding_dim
        self.state_size = state_size
        
        # 创建 actor 网络和目标网络
        self.network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network = ActorNetwork(embedding_dim, hidden_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.tau = tau
    
    def build_networks(self):
        # 触发网络构建
        self.network(np.zeros((1, 3 * self.embedding_dim)))
        self.target_network(np.zeros((1, 3 * self.embedding_dim)))
    
    def update_target_network(self):
        c_theta, t_theta = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(c_theta)):
            t_theta[i] = self.tau * c_theta[i] + (1 - self.tau) * t_theta[i]
        self.target_network.set_weights(t_theta)
        
    def train(self, states, dq_das):
        with tf.GradientTape() as g:
            outputs = self.network(states)
        dj_dtheta = g.gradient(outputs, self.network.trainable_weights, -dq_das)
        grads = zip(dj_dtheta, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        
    def save_weights(self, path):
        self.network.save_weights(path)
        
    def load_weights(self, path):
        self.network.load_weights(path)