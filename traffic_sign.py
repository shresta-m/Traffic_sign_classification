import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data_set = []
class_labels = []
total_classes = 43
curr_path = os.getcwd()

for i in range(total_classes):
	path = os.path.join(curr_path,'train',str(i))
	images = os.listdir(path)

	for j in images:
		try:
			image = Image.open(path + '/' + j)
			image = image.resize((30,30))
			image = np.array(image)
			data_set.append(image)
			class_labels.append(i)
		except:
			print("Error loading the image")

data_set = np.array(data_set)
class_labels = np.array(class_labels)
# print(len(data_set),len(class_labels))

x_train,x_test,y_train,y_test = train_test_split(data_set,class_labels,test_size = 0.3, random_state = 42)
print(x_train.shape,x_test.shape)

y_train = to_categorical(y_train,43)
y_test = to_categorical(y_test,43)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_test, y_test))
model.save("trained_model.h5")

plt.figure(0)
plt.plot(history.history['accuracy'],label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'],label = 'val accuracy')
plt.title('ACCURACY')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'],label = 'Training loss')
plt.plot(history.history['val_loss'],label = 'val loss')
plt.title('LOSS')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_test = pd.read_csv('Test.csv')
class_labels = y_test['ClassId'].values
images_data = y_test['Path'].values

data = []
for x in images_data:
	image = Image.open(x)
	image = image.resize((30,30))
	data.append(np.array(image))

x_test = np.array(data)
predicted_labels = model.predict_classes(x_test)
print(accuracy_score(class_labels,predicted_labels))