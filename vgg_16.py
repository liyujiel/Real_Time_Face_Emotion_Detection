from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2


def vgg16_model(img_shape=(224, 224, 3), n_classes=1000, l2_reg=0.,
	weights=None):

	# Initialize model
	vgg16 = Sequential()

	# Layer 1 & 2
	vgg16.add(Conv2D(64, (3, 3), padding='same',
		input_shape=img_shape, kernel_regularizer=l2(l2_reg)))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(64, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3 & 4
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(128, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(128, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 5, 6, & 7
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(256, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(256, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(256, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 8, 9, & 10
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 11, 12, & 13
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 14, 15, & 16
	vgg16.add(Flatten())
	vgg16.add(Dense(4096))
	vgg16.add(Activation('relu'))
	vgg16.add(Dropout(0.5))
	vgg16.add(Dense(4096))
	vgg16.add(Activation('relu'))
	vgg16.add(Dropout(0.5))
	vgg16.add(Dense(n_classes))
	vgg16.add(Activation('softmax'))

	if weights is not None:
		vgg16.load_weights(weights)

	return vgg16
