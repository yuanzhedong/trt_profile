import tensorflow as tf
from utils import config_gpu_memory
from seg_dataset import SegmentationDataset
from tensorflow.python.tools import freeze_graph
import os
import shutil

def FCN_model(len_classes=2, dropout_rate=0.2):

    inputs = tf.keras.layers.Input(shape=(512, 512, 3))
    c1 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal',padding='same')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation("relu")(c1)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    
    c2 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation("relu")(c2)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)


    c3 = tf.keras.layers.Conv2D(64, (3, 3),  kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation("relu")(c3)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation("relu")(c4)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation("relu")(c5)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Activation("relu")(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = tf.keras.layers.concatenate([u7, c3])

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = tf.keras.layers.concatenate([u8, c2])

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = tf.keras.layers.concatenate([u9, c1])

    c9 = tf.keras.layers.BatchNormalization()(u9)
    c9 = tf.keras.layers.Activation("relu")(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    print(model.summary())
    return model

def save_v2(model):
    model.save_weights("tmp_tf_weights.h5")
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)

    model = FCN_model()
    model.load_weights("tmp_tf_weights.h5")
    save_dir = "./fcn_model/"

    shutil.rmtree(save_dir) if os.path.exists(save_dir) else None
    tf.saved_model.simple_save(tf.keras.backend.get_session(),
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
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.keras.backend.set_session(tf.Session(config=config));
    model = FCN_model(len_classes=1, dropout_rate=0.2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dataset = SegmentationDataset(size = 512, block_size = 64)
    x_train, y_train = dataset.get_data(8)
    x_val, y_val = dataset.get_data(4)
    model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=1, validation_data = (x_val, y_val))
    save_v2(model)
