'''Plan:
1. Building a data pipeline
2. preprocessing images for DL
3. creating a deep NN classifier
4. Evaluating model performance'''

##Building a data pipeline

'''Setup and Load data'''
#1.1 Install dependencies amd setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt

#1.2 Remove Dodgy images

import cv2
import imghdr

data_dir = 'datasets'
image_exts = ['jpeg','jpg','png','bmp']
'''img = cv2.imread(os.path.join('datasets','happy','smile.woman_.jpg'))
plt.imshow(img)
plt.show()'''

for image_class in os.listdir(data_dir):
# first we are looping through every folder that we've got inside our datasets directory, so it should print the names of happy and sad folder
    for image in os.listdir(os.path.join(data_dir,image_class)):
# Now, we'll go through every single image inside of those subdirectories
        image_path = os.path.join(data_dir,image_class,image)
        try:
#just double checking that 1) we can load the images in opencv and 2) that our image matches the path or exts
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
#even after double checking if its still wierd we're going to remove it from that folder
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with Image {}'.format(image_path))


#1.3 Load data
import numpy as np
#lets load our data
#this is basically building our data pipeline
data = tf.keras.utils.image_dataset_from_directory('datasets',batch_size=15)


#I cant look at this data or just grab the first instance because this is'nt a dataset which is preloaded in the memory already, it is actually a generator
#for that lets connvert it into an iterator

#this allowing us to access our data pipeline
data_iterator = data.as_numpy_iterator()

#accessing the data pipeline

batch = data_iterator.next()
'''print(batch[0])#images
print(batch[1])#labels'''



'''Preprocess data'''

#2.1 Scale the data

data = data.map(lambda  x, y:(x/255,y)) #data.map allows you to perform that transformation in pipeline and lambda is used to go and do that transformation

#2.2 Split the data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(batch[0],batch[1],test_size=0.3)

'''Building Deep model'''
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

#3.1 Build a deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(256,256,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')

])
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

#3.2

logdir = 'logs'
callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(x_train,y_train,epochs=10,callbacks=callback)

#3.3 plot performace
fig = plt.figure()
plt.plot(hist.history['loss'],color = 'teal',label = 'loss')
plt.title('loss',fontsize = 15)
plt.show()


#Evaluate the model
y_pred = model.predict(x_test).reshape(-1)
y_pred=np.round(y_pred)

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,y_pred))


