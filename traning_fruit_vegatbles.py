#libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#training  image processing
training_set = tf.keras.utils.image_dataset_from_directory(
    "train",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64,64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    
)


# Validation image processing
validation_set = tf.keras.utils.image_dataset_from_directory(
    "validation",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

#Building Model

cnn = tf.keras.models.Sequential()
     
#Building Convolution Layer

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides = 2))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides = 2))    
cnn.add(tf.keras.layers.Dropout(0.5))  #to avoid overfitting
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=512,activation="relu"))
cnn.add(tf.keras.layers.Dense(units=256,activation="relu"))

#we are dropping some neuron to avoid  overfiting
cnn.add(tf.keras.layers.Dropout(0.5))

#Output Layer
cnn.add(tf.keras.layers.Dense(units=36,activation='softmax'))
     
#Compiling and Training Phase
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])     
cnn.summary()


training_history = cnn.fit(x=training_set,validation_data=validation_set,epochs=30)


#Saving Model

cnn.save('trained_model.h5')
print(training_history.history) #Return Dictionary of history

#Recording History in json
import json
with open('training_hist.json','w') as f:
  json.dump(training_history.history,f)
     

print(training_history.history.keys())
     

#Calculating Accuracy of Model Achieved on Validation set
print("Validation set Accuracy: {} %".format(training_history.history['val_accuracy'][-1]*100))


epochs = [i for i in range(1,31)]
plt.plot(epochs, training_history.history['accuracy'],color='red')
plt.xlabel("NO, of Epochs")
plt.ylabel('Traning Accuracy')
plt.title("Visualization Of Traning Accuracy Results")
plt.show()


plt.plot(epochs,training_history.history['val_accuracy'],color ='blue')
plt.xlabel("No, of Epochs")
plt.ylabel('Validation Accuaracy')
plt.title("Visualization Of Validation Accuracy Results")
plt.show()

#Evaluation Models
training_loss,training_accuracy = cnn.evaluate(training_set)
print(training_loss,training_accuracy)

valiadtion_loss,validation_accuracy = cnn.evaluate(validation_set)
print(valiadtion_loss,validation_accuracy)

#Test set Evaluation 
test_set = tf.keras.utils.image_dataset_from_directory(
    "test",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64,64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    
)
test_loss,test_accuracy = cnn.evalute(test_set)
print(test_loss,test_accuracy)

