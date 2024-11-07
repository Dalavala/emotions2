import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2D, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import numpy as np

@tf.keras.utils.register_keras_serializable()
class Model(tf.keras.Model):
    def __init__(self, base_model, name=None):
      super(Model, self).__init__(name=name)
      self.base_model = base_model

      # Добавляем другие слои
      self.global_average_pooling = GlobalAveragePooling2D()
      self.dense1 = Dense(512, activation='relu')
      self.batch_norm = tf.keras.layers.BatchNormalization()
      self.dropout = tf.keras.layers.Dropout(0.2)
      self.dense2 = Dense(10, activation='sigmoid')

    def get_loss(self, y, preds):
        return tf.keras.losses.CategoricalCrossentropy()(y, preds)

    @tf.function
    def training_step(self, x, y):
        with tf.GradientTape() as tape:
            preds = self.call(x)  # Вызываем call() для получения предсказаний
            loss = self.get_loss(y, preds)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return tf.reduce_mean(loss)

    def call(self, inputs, training=True):
        # Передаем данные через базовую модель и добавленные слои
        x = self.base_model(inputs, training=training)
        x = self.global_average_pooling(x)
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        return self.dense2(x)

    def get_config(self):
        config = super().get_config()
        config['base_model'] = self.base_model.to_json()
        config['global_average_pooling'] = self.global_average_pooling.get_config()
        config['dense1'] = self.dense1.get_config()
        config['batch_norm'] = self.batch_norm.get_config()
        config['dropout'] = self.dropout.get_config()
        config['dense2'] = self.dense2.get_config()
        return config

    @classmethod
    def from_config(cls, config):
        base_model = tf.keras.models.model_from_json(config['base_model'])
        model = cls(base_model)
        model.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D.from_config(config['global_average_pooling'])
        model.dense1 = Dense.from_config(config['dense1'])
        model.batch_norm = tf.keras.layers.BatchNormalization.from_config(config['batch_norm'])
        model.dropout = tf.keras.layers.Dropout.from_config(config['dropout'])
        model.dense2 = Dense.from_config(config['dense2'])
        return model