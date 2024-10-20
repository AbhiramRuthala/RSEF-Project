import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import colorama
from colorama import Fore, Back
import dlib
import imghdr

from keras.src.metrics.accuracy_metrics import accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, BinaryAccuracy, Recall
import pync

import time
import plyer
from plyer import notification
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics

video = cv2.VideoCapture(0)

detection = dlib.get_frontal_face_detector()
prediction = dlib.shape_predictor("/Users/abhiramruthala/Downloads/shape_predictor_68_face_landmarks.dat")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detection(gray, 0)

    for face in faces:
        # Get the facial landmarks for the face
        landmarks = prediction(gray, face)


        # Loop over the facial landmarks
        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

    cv2.imshow("Frame", frame)

    faceszn = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)

    for (x,y,w,h) in faceszn:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    if cv2.waitKey(1) == ord("s"):
        cv2.imwrite('abhiram.jpg', frame)
        break

#    if cv2.waitKey(1) == ord("q"):
#        break

video.release()
cv2.destroyAllWindows()

print("Let's proceed by having the model evaluate you.")


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Remove bad/corrupted images
datadir = 'datas'
imgext = ['jpeg', 'png', 'jpg', 'bmp']

print(os.listdir(datadir))
for imgclass in os.listdir(datadir):
    for image in os.listdir(os.path.join(datadir, imgclass)):
        image_path = os.path.join(datadir, imgclass, image)
        try:
            gang = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in imgext:
                print("Image ain't there gang {}".format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("Corrupted image {}".format(image_path))

dataszn = tf.keras.utils.image_dataset_from_directory('datas')
dataiterator = dataszn.as_numpy_iterator()
batch = dataiterator.next()

#Happy = 1
#Angry = 0
#sad = 2
#Anxiety = 3
#fig, ax = plt.subplots(ncols=6, figsize=(10, 10))
#for idx, img in enumerate(batch[0][:6]):
#    ax[idx].imshow(img.astype(int))
#    ax[idx].title.set_text(batch[1][idx])
#plt.show()

#scale = batch[0] / 255
dataszn = dataszn.map(lambda x, y: (x/255,y))
newbatch = dataszn.as_numpy_iterator().next()

#training and validation datasets are used during training, whilst the test dataset is used during the testing phase. The model is mainly trained on the training data, and is finetuned using the validation data.
train_data = int(len(dataszn)*0.7)
test_data = int(len(dataszn)*0.1)+2
val_data = int(len(dataszn)*0.15)+1

train = dataszn.take(train_data)
test = dataszn.skip(train_data).take(test_data)
val = dataszn.skip(train_data+test_data).take(val_data)

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy, metrics=['accuracy'])

logsdir = 'logs'

tensorcallback = tf.keras.callbacks.TensorBoard(log_dir = logsdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorcallback])

#figureszn = plt.figure()

#plt.plot(hist.history['accuracy'], color='red', label='accuracy')
#plt.plot(hist.history['loss'], color='blue', label='loss')
#figureszn.suptitle('Accuracy vs. Loss in DNN model', fontsize=18)
#plt.legend(loc="upper left")
#plt.xlabel('Time')
#plt.ylabel('Value')
#plt.show()

pre = Precision()
acc = BinaryAccuracy()
rec = Recall()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    acc.update_state(y, yhat)
    rec.update_state(y, yhat)

#print(f'Precision: {pre.result()}, Accuracy: {acc.result()}, Recall: {rec.result()}')

img = cv2.imread('abhiram.jpg')

rezimg = tf.image.resize(img, (256, 256))
plt.imshow(rezimg.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(rezimg/255, 0))
print(Fore.RED + yhat)

#Mood tracking system
moodtype = "No mood data to track"

if yhat > 0.5:
    #yellow = cv2.imread('yellow.webp')
    #rezimg = cv2.cvtColor(yellow, cv2.COLOR_BGR2RGB)
    #plt.imshow(rezimg)
    #plt.show()
    notification.notify(
        title="Hey there!",
        message="You're doing well! You're going to be fine.",
        app_name="Stability",

        timeout=5
    )
    moodtype = "happy"
    #print("Your mood seems pretty good, and I've just noticed that your friends want to go out. Would you like me to respond saying YES?")
    #print("Dude an adaptable text reader that gives you clear-cut notis based on your mood would be nice af.")
else:
    pync.notify("Aw man! I'm sorry you're feeling that way.")
    pync.notify("You should stay away from things that make you angry.")
    pync.notify("Channel your energy into more important activities that you can do in the day.")

    notification.notify(
        title="Hey there!",
        message="I'm sorry that you're feeling angry. Go for a walk to help rehabilitate yourself.",
        app_name="Stability",

        timeout=5
    )

    moodtype = "angry"
    #print("Should be an angry person")
    #blue = cv2.imread('blue.jpg')
    #rezimg2 = cv2.cvtColor(blue, cv2.COLOR_BGR2RGB)
    #plt.imshow(rezimg2)
    #plt.show()
    #print("I suggest you go out with your friends")


def MoodTrack():
    global moodtype
    if moodtype == "happy":
        notification.notify(
            title="You have started a streak!",
            message="Good job! You've just started a streak of being happy.",
            app_name="Stability",

            timeout=5
        )

    elif moodtype == "angry":
        notification.notify(
            title="Oh no",
            message="You were feeling angry today. Let's change that.",
            app_name="Stability",

            timeout=5
        )

MoodTrack()
