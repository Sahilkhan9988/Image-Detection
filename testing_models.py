import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as  plt

#Loading Models 
cnn = tf.keras.models.load_model("trained_model.h5")


import cv2
# Load and preprocess the test image using TensorFlow
image_path = r"C:\Users\kajan computers wari\Desktop\project1\test\watermelon\Image_1.jpg"
img = cv2.imread(image_path)

plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show() 

# Load the image using TensorFlow's image loading utilities
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))

# Convert the image to a NumPy array and normalize pixel values to be between 0 and 1
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr]) # converting single  image to batch

# Make predictions using the pre-trained model
predictions = cnn.predict(input_arr)




# Get the predicted class index
print(predictions[0])
print(max(predictions[0]))



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

print(test_set.class_names)
# Get the predicted class label using the class names from the test set
result_index = np.where(predictions[0] == max(predictions[0]))

print(result_index[0][0])

#display image 
plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show() 


#single predictions 
print("It's a {}".format(test_set.class_names[result_index[0][0]]))