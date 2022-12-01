import matplotlib.pyplot as plt

#------------------------------------------------------
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions

# For tensorflow.keras
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.preprocessing.image import load_img
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.applications.vgg16 import decode_predictions
#------------------------------------------------------

model = VGG16('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
print(model.summary())

# load an image from file
image = load_img('cat.jpg', target_size=(224, 224))
plt.imshow(image)
plt.show()

# convert the image pixels to a numpy array
image = img_to_array(image)
print(image.shape)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
print(image.shape)

y_pred = model.predict(image)
print(y_pred.shape)

# convert the probabilities to class labels
labels_pred = decode_predictions(y_pred)
print(labels_pred)

