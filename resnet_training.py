import tensorflow as tf
import traceback
import gc
import utils
import time
from keras.layers import merge
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.callbacks import TensorBoard
from sklearn.cross_validation import train_test_split
from keras import backend as K


def identity_block(x, nb_filter, kernel_size=3):
    k1, k2, k3 = nb_filter

    out = Conv2D(k1, 1, 1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(k2, kernel_size, kernel_size, border_mode='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(k3, 1, 1)(out)
    out = BatchNormalization()(out)

    out = merge([out, x], mode='sum')
    out = Activation('relu')(out)

    return out


def conv_block(x, nb_filter, stage, kernel_size=3):
    k1, k2, k3 = nb_filter

    # 大于2时主路径取subsample
    if stage > 2:
        out = Conv2D(k1, 1, 1, subsample=(2, 2))(x)
    else:
        out = Conv2D(k1, 1, 1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = out = Conv2D(k2, kernel_size, kernel_size)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(k3, 1, 1)(out)
    out = BatchNormalization()(out)

    # 大于2时shortcut取subsample
    if stage > 2:
        x = Conv2D(k3, 1, 1, subsample=(2, 2))(x)
    else:
        x = Conv2D(k3, 1, 1)(x)
    x = BatchNormalization()(x)

    # 调整大小
    out = ZeroPadding2D((1, 1))(out)

    #     print 'out shape = ' + str(out.shape)
    #     print 'x shape = ' + str(x.shape)

    out = merge([out, x], mode='sum')
    out = Activation('relu')(out)

    return out


def gen_resnet_model():
    inp = Input(shape=(1, 1050, 1680))
    #     print 'Input shape = ' + str(inp.shape)
    out = ZeroPadding2D((3, 3))(inp)
    out = Conv2D(64, 7, 7, subsample=(2, 2))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPooling2D((3, 3), strides=(2, 2))(out)

    out = conv_block(out, [64, 64, 256], 2)
    out = identity_block(out, [64, 64, 256])
    out = identity_block(out, [64, 64, 256])

    out = conv_block(out, [128, 128, 512], 3)
    out = identity_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])

    out = conv_block(out, [256, 256, 1024], 4)
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])

    out = conv_block(out, [512, 512, 2048], 5)
    out = identity_block(out, [512, 512, 2048])
    out = identity_block(out, [512, 512, 2048])
    print 'Before Average Pooling Shape = ' + str(out.shape)

    # (33, 53) 配合上面的结构
    out = AveragePooling2D((33, 53))(out)
    print 'After Average Pooling Shape = ' + str(out.shape)
    out = Reshape((2048,))(out)
    print 'After Reshape = ' + str(out.shape)
    #     out = Dense(5, input_dim=(2048,), activation='softmax')(out)
    out = Dense(5, activation='softmax')(out)
    print 'Last Output = ' + str(out.shape)

    model = Model(inp, out)

    return model


def train_resnet(output_path, X, label, log_path):
    max_acc = 0
    for cnt in range(3):
        with tf.device('/gpu:1'):
            print '\nStarting Round ' + str(cnt) + ' at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            model = gen_resnet_model()
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()

            # 随机划分训练集和测试集
            x_train, x_test, y_train, y_test = train_test_split(X, label)

            for i in range(3):
                print '\nStarting Round ' + str(cnt) + ' && i = ' + str(i) + ' at ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                                    time.localtime()) + '\n'

                his = model.fit(x_train, y_train, batch_size=1, epochs=1, validation_split=0.1, verbose=1,
                                callbacks=[TensorBoard(log_dir=log_path)])
                print his
                score = model.evaluate(x_test, y_test, batch_size=1, verbose=1)
                print score

                if score[1] > max_acc:
                    max_acc = score[1]
                    model.save(output_path + 'max_acc.model')

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
    original_stdout = utils.before('resnet_training_console_output')
    try:
        X, label = utils.gen_data('../../data/whole/sharpened/whole_5_classes.h5')
        train_resnet('../model/resnet', X, label, './')
    # gen_resnet_model()
    except Exception, e:
        print e
        traceback.print_exc()
        raise
    finally:
        utils.after(original_stdout)
