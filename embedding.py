import tensorflow as tf
import numpy as np

class MovieGenreEmbedding(tf.keras.Model):
    def __init__(self, len_movies, len_genres, embedding_dim):
        super(MovieGenreEmbedding, self).__init__()
        # 直接创建各个层，不需要用 InputLayer 包装
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        self.g_embedding = tf.keras.layers.Embedding(name='genre_embedding', input_dim=len_genres, output_dim=embedding_dim)
        self.m_g_merge = tf.keras.layers.Dot(name='movie_genre_dot', normalize=True, axes=1)
        self.m_g_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        # 假定 inputs 是一个包含两个张量的列表或元组：[movie_ids, genre_ids]
        movie_ids, genre_ids = inputs
        memb = self.m_embedding(movie_ids)
        gemb = self.g_embedding(genre_ids)
        m_g = self.m_g_merge([memb, gemb])
        return self.m_g_fc(m_g)


class UserMovieEmbedding(tf.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbedding, self).__init__()
        # 同样不使用 InputLayer
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        # 假定 inputs 是一个包含两个张量的列表或元组：[user_ids, movie_ids]
        user_ids, movie_ids = inputs
        uemb = self.u_embedding(user_ids)
        memb = self.m_embedding(movie_ids)
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)