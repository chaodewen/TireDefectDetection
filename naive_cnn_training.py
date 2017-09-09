import tensorflow as tf
import traceback
import gc
import time
import utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.cross_validation import train_test_split
from keras.callbacks import TensorBoard
from keras import backend as K


def train_naive_cnn(output_path, X, label, log_path):
    max_acc = 0

    for cnt in range(5):
        with tf.device('/gpu:1'):
            print '\nStarting Round ' + str(cnt) + ' at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            model = Sequential()
            model.add(Conv2D(32, (5, 5), input_shape=(1, 1050, 1680), activation='relu'))
            model.add(Conv2D(32, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(32, kernel_initializer='normal', activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(5, kernel_initializer='normal', activation='softmax'))
            model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()

            # 随机划分训练集和测试集
            x_train, x_test, y_train, y_test = train_test_split(X, label)

            for i in range(5):
                print '\nStarting Round ' + str(cnt) + ' && i = ' + str(i) + ' at ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                                    time.localtime()) + '\n'

                his = model.fit(x_train, y_train, batch_size=1, epochs=1, validation_split=0.1, verbose=1,
                                callbacks=[TensorBoard(log_dir=log_path)])
                print his
                score = model.evaluate(x_test, y_test, batch_size=1, verbose=1)
                print score

                if score[1] > max_acc:
                    max_acc = score[1]
                    model.save(output_path + 'whole_naive_cnn.model')

                print '\nEnding Round ' + str(cnt) + ' && i = ' + str(i) + ' at ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                                  time.localtime())

            # 释放内存 否则会耗尽资源
            del x_train
            del x_test
            del y_train
            del y_test
            del his
            del score
            del model
            # 真正释放
            gc.collect()
            # 释放Keras
            K.clear_session()

            print '\nEnding Round ' + str(cnt) + ' at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


if __name__ == "__main__":

    original_stdout = utils.before('naive_cnn_training_console_output')

    try:
        X, label = utils.gen_data('../../data/whole/sharpened/whole_5_classes.h5')
        train_naive_cnn('../model/naive_cnn', X, label, './')
    except Exception, e:
        print e
        traceback.print_exc()
        raise
    finally:
        utils.after(original_stdout)