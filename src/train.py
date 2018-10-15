from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils


def data_generator():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    while True:
        for i in range(1875):  # 1875 * 32 = 60000 -> # of training samples
            if i % 125 == 0:
                print("i = {}".format(str(i)))
            # (32, 1, 28, 28), (32, 10)
            yield X_train[i * 32:(i + 1) * 32], y_train[i * 32:(i + 1) * 32]


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=[28, 28, 1]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer='adadelta',
    metrics=['accuracy']
)

model.fit_generator(
    data_generator(), steps_per_epoch=60000 // 32, nb_epoch=10,
    validation_data=None
)
model.save_weights('../models/best_weight.h5')
json_string = model.to_json()

with open('../models/config.json', 'w') as f:
    f.write(json_string)
