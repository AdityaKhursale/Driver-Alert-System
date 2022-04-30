import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name='gpu')

# TODO: move below variables into src/shared_data.py

TRAIN_DIR = 'datasets/fer2013/train'
TEST_DIR = 'datasets/fer2013/test'
BATCH_SIZE = 64
NUM_EPOCH = 50
NUM_TRAIN = 28709
NUM_TEST = 7178
LOSS = 'categorical_crossentropy'
LEARNING_RATE = 0.0001
DECAY = 1e-6
TARGET_SIZE = (48,48)

# TODO: implemented industry standard object oriented practices.

class Emotion:
    
    def __init__(self) -> None:
        pass
    
    def data_generator(self):
        train_data_generator = ImageDataGenerator(rescale=1./255)
        test_data_generator = ImageDataGenerator(rescale=1./255)

        train_generator = train_data_generator.flow_from_directory(TRAIN_DIR,
                                                                target_size=TARGET_SIZE,
                                                                batch_size=BATCH_SIZE,
                                                                color_mode='grayscale',
                                                                class_mode='categorical')

        test_generator = test_data_generator.flow_from_directory(TEST_DIR,
                                                                target_size=TARGET_SIZE,
                                                            batch_size=BATCH_SIZE,
                                                            color_mode='grayscale',
                                                            class_mode='categorical')

        return (train_generator, test_generator)


    

if __name__ == '__main__':
    emotion = Emotion()
    train_generator, test_generator = emotion.data_generator()
    

    
    
    
