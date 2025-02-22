import tensorflow as tf
import sounddevice
from keras.src.metrics.accuracy_metrics import accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt


#Not sure if we need these yet because we are only looking for the accuracy metric. It doesn't hurt to have other metrics though.
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

print("Let's start by having the model evaluate your speech")

name = input("What's your name? ")

#Speech related components come here
#This would involve the use of sounddevice, microphone etc.

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#Data collection/selection points.

#Separate data into batches

#Data analysis points from the batches.


#train_data
#test_data
#val_data

#Need configuration of values to see which value represents Alzheimer's Disease or Normal voice (1 or 0)

#Skip certain parts of the data.

#train = dataszn.take(train_data)
#test = dataszn.skip(train_data).take(test_data)
#val = dataszn.skip(train_data+test_data).take(val_data)


model = Sequential()

model.add()

model.add(Flatten())

#Go from 256 neurons to 1 neuron.
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy, metrics=['accuracy'])

#Run lines of code that provides the result of the data.

#history = model.fit()

#Plot the accuracy graph and display it at the end of the code.

print(f"Here are the results of {name}'s examination with this tool.")

print(f"Thanks for using it {name}")
