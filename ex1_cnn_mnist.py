import matplotlib.pyplot as plt
import matplotlib.cm as cm

#--------------------------------------------------------
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

# For tensorflow.keras
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.layers import Dropout
#from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import Dense
#--------------------------------------------------------

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print( X_train.shape )
plt.imshow(X_train[0], cmap = cm.Greys)
plt.show()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_train /= 255
print(X_train.shape)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test = X_test.astype('float32')
X_test /= 255

print(y_train.shape)
print(y_train[0:3])

#---------------------------------------------
y_train = np_utils.to_categorical(y_train, 10)

# For tensorflow.keras
#y_train = to_categorical(y_train, 10)
#---------------------------------------------

print(y_train[0])

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
print(model.output_shape)

model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)

model.add(Dropout(0.25))
print(model.output_shape)

model.add(Flatten())
print(model.output_shape)

model.add(Dense(128, activation='relu'))
print(model.output_shape)

model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.output_shape)

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
                                           
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

plt.imshow(X_test[0].reshape(28,28), cmap = cm.Greys)
plt.show()

print(model.predict(X_test[0].reshape(1, 28, 28, 1)))

