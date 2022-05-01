from ast import arg
import time
import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from imutils.video import VideoStream
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
MODEL_WT_DIR = 'models/weights'

args = None 

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
    
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
  
        return model

    def compile_model(self, model):
        optimizer = Adam(learning_rate=LEARNING_RATE,decay=DECAY)
        metrics = ['accuracy']
        model.compile(loss=LOSS,optimizer=optimizer,metrics=metrics)
        return model

    def train_model(self, model):
        model_info = model.fit_generator(train_generator,
                                        steps_per_epoch=NUM_TRAIN // BATCH_SIZE,
                                        epochs=NUM_EPOCH,
                                        validation_data=test_generator,
                                        validation_steps=NUM_TEST // BATCH_SIZE)
        return model_info


def get_video_input(model):
    # prevents openCL usage and unnecessary logging messages
    # cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 
                    5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = VideoStream(0).start()
    time.sleep(1.0)

    while True:
        # Find haar cascade to draw bounding box around face
        # ret, frame = cap.read() # cv2 function
        frame = cap.read()
        # if not ret:
        #     break
        facecasc = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.stop()   

def populate_arguments():
    global args
    parser = argparse.ArgumentParser(description="Emotion detection")
    parser.add_argument("--logsdir", required=False, default=".",
                        help="directory to store logs")
    parser.add_argument("--mode", required=False, default='load',
                        help="train or load the model")
    args = parser.parse_args()
    

if __name__ == '__main__':
    
    populate_arguments()
    
    emotion = Emotion()
    train_generator, test_generator = emotion.data_generator()
    model = emotion.create_model()
    model = emotion.compile_model(model)
    # print(model.summary())
    if args.mode == "train":
        trained_model = emotion.train_model(model)
    model.load_weights(os.path.join(MODEL_WT_DIR, 'simple_model_v1.h5'))
    get_video_input(model)
    
    

    
    
    
