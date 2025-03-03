import tensorflow as tf
import librosa
import numpy as np
import os
import sounddevice
from coremltools.proto.NeuralNetwork_pb2 import AdamOptimizer
from keras.src.metrics.accuracy_metrics import accuracy
from nipype.testing import filepath
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
import matplotlib.pyplot as plt

print("Let's start by having the model evaluate your speech")

name = input("What's your name? ")

#Speech related components come here
#This would involve the use of sounddevice, microphone etc.

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'trainingData'
classes = ['Alzheimer-Speech', "Normal-Speech"]

def preprocessData(data_dir, classes, target_shape=(128,128)):
    data = []
    labels = []

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                filepathszn = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(filepathszn, sr=None)

                mel_spectrogram = librosa.feature.mel_spectrogram(y=audio_data, sr=sample_rate)
                mel_spectrogram = tf.image.resize(np.expand_dims(audio_data, axis=-1), target_shape)
                data.append(mel_spectrogram)
                labels.append(i)

            return np.array(data), np.array(labels)

data, labels = preprocessData(data_dir, classes)
labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes))
X_train, X_test, Y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

input_shape = X_train[0].shape
input_layer = Input(shape = input_shape)
x= Conv2D(32, (3, 3), activation='relu')(input_layer)
x=MaxPooling2D((2,2))(x)
x=Conv2D(64, (3,3 ), activation='relu')(x)
x= MaxPooling2D((2,2))(x)
x= Flatten()(x)
x= Dense(64, activation='relu')(x)
output_layer = Dense(len(classes), activation='softmax')(x)
model = Model(input_layer, output_layer)

model.compile(optimizer=AdamOptimizer(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

Accuracy = model.evaluate(X_test, y_test, verbose=0)
print(Accuracy[1])

model.save('AlzSpeechMLmodel.h5')

print(f"Here are the results of {name}'s examination with this tool.")

