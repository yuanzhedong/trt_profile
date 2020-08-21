import tensorflow as tf
from utils import config_gpu_memory
from seg_dataset import SegmentationDataset
from tensorflow.python.tools import freeze_graph
import os
import shutil

def FCN_model(len_classes=2, dropout_rate=0.2):
    
    input = tf.keras.layers.Input(shape=(512, 512, 3))

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    predictions = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')

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
    dataset = SegmentationDataset(size = 512, block_size = 64)
    x_train, y_train = dataset.get_data(8)
    x_val, y_val = dataset.get_data(4)
    model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=1, validation_data = (x_val, y_val))
    save_v2(model)
