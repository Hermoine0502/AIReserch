import numpy as np

#------------------------------------------------------------------------------------------
#from keras.datasets import mnist
#from keras.utils import np_utils
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
#from keras.optimizers import Adam
#from keras.models import load_model

# For tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
#------------------------------------------------------------------------------------------

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.

#---------------------------------------------------------
#y_train = np_utils.to_categorical(y_train, num_classes=10)
#y_test = np_utils.to_categorical(y_test, num_classes=10)

# For tensorflow.keras
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
#---------------------------------------------------------

model = load_model('CNN_example_model.h5')

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

