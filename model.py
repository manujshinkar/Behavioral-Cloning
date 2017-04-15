import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(img):
	new_img = img[50:140,:,:]
	new_img = cv2.resize(img,(200, 66), interpolation = cv2.INTER_AREA)
	return new_img

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
steering_correction = 0.25
for line in lines[1:]:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'data/IMG/' + filename
		image = cv2.imread(current_path)
		#processed_image = preprocess_image(image)
		images.append(image)
		measurement = float(line[3])
		if i == 1:
			 measurement += steering_correction
		elif i == 2:
			 measurement -= steering_correction
		measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)
	
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#plt.figure(figsize=(160, 320))
#plt.imshow(images[4])
#plt.show()

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda 
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
#from keras.layers.cropping import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5,subsample = (2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5,subsample = (2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5,subsample = (2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



'''
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5') 
