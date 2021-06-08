import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.python.tools import freeze_graph
import os
import shutil

input_size = 512
def process_dataset():
    # Import the data
    NUM_TRAIN = 10
    #x_train = np.random.randint(10, size=NUM_TRAIN)
    x_train = np.ones((NUM_TRAIN , input_size, input_size, 1))
    y_train = x_train
    x_test = x_train
    y_test = x_train
    x_train = np.reshape(x_train, (NUM_TRAIN, input_size, input_size, 1))
    x_test = np.reshape(x_test, (NUM_TRAIN, input_size, input_size, 1))
    return x_train, y_train, x_test, y_test

class DepthMaxPool(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding="VALID", **kwargs):
        super().__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
    def call(self, inputs):
        return tf.nn.max_pool(inputs,
                              ksize=(1, 1, 1, self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)
def create_model():
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.InputLayer(input_shape=[input_size, input_size,1]))
    # model.add(tf.keras.layers.Activation("relu"))
    # model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    # return model

    inputs = tf.keras.layers.Input((input_size, input_size,1))
    c1 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(inputs)
    c2 = tf.keras.layers.Activation("relu")(c1)
    #c2 = tf.keras.layers.concatenate([c2, c1], axis=3)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c2)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    return model

def save_v2(model, filename):
    model.save_weights("tmp_tinynet_weights.h5")
    K.clear_session()
    K.set_learning_phase(0)

    model = create_model()
    model.load_weights("tmp_tinynet_weights.h5")

    save_dir = "tmp_tinynet"
    shutil.rmtree(save_dir) if os.path.exists(save_dir) else None
    # import pdb
    # pdb.set_trace()
    tf.saved_model.simple_save(K.get_session(),
                           save_dir,
                           inputs={"input": model.inputs[0]},
                           outputs={"output": model.outputs[0]})

    freeze_graph.freeze_graph(None,
                              None,
                              None,
                              None,
                              model.outputs[0].op.name,
                              None,
                              None,
                              os.path.join(save_dir, "frozen_model.pb"),
                              False,
                              "",
                              input_saved_model_dir=save_dir)
def main():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.keras.backend.set_session(tf.Session(config=config));

    x_train, y_train, x_test, y_test = process_dataset()
    # import pdb
    # pdb.set_trace()
    model = create_model()
    # Train the model on the data
    model.fit(x_train, y_train, epochs = 2, verbose = 1)
    # Evaluate the model on test data
    model.evaluate(x_test, y_test)
    save_v2(model, filename="models/lenet5.pb")

if __name__ == '__main__':
    main()

