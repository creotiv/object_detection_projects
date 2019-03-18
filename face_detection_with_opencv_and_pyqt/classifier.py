import logging
import sys
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from kito import reduce_keras_model
import matplotlib.pyplot as plt
import tensorflow as tf
sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"

        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    # Wide residual network http://arxiv.org/abs/1605.07146
    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
            # format of conv_params:
            #               [ [kernel_size=("kernel width", "kernel height"),
            #               strides="(stride_vertical,stride_horizontal)",
            #               padding="same" or "valid"] ]
            # B(3,3): orignal <<basic>> block
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)

                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                          strides=v[2],
                                          padding=v[3],
                                          kernel_initializer=self._weight_init,
                                          kernel_regularizer=l2(self._weight_decay),
                                          use_bias=self._use_bias)(convs)
                else:
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                          strides=v[2],
                                          padding=v[3],
                                          kernel_initializer=self._weight_init,
                                          kernel_regularizer=l2(self._weight_decay),
                                          use_bias=self._use_bias)(convs)

            # Shortcut Connection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(n_output_plane, kernel_size=(1, 1),
                                         strides=stride,
                                         padding="same",
                                         kernel_initializer=self._weight_init,
                                         kernel_regularizer=l2(self._weight_decay),
                                         use_bias=self._use_bias)(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f


    # "Stacking Residual Units on the same stage"
    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    # def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        conv1 = Conv2D(filters=n_stages[0], kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              kernel_initializer=self._weight_init,
                              kernel_regularizer=l2(self._weight_decay),
                              use_bias=self._use_bias)(inputs)  # "One conv at the beginning (spatial size: 32x32)"

        # Add wide residual blocks
        block_fn = self._wide_basic
        conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(conv1)
        conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(conv2)
        conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(conv3)
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        # Classifier block
        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
        flatten = Flatten()(pool)
        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",
                              name="pred_gender")(flatten)
        predictions_a = Dense(units=101, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",
                              name="pred_age")(flatten)
        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])

        return model

class FaceClass:
    
    def __init__(self,img_size=64, depth=16, width=8, pretrained_model = "weights.29-3.76_utk.hdf5"):
        self.img_size = img_size
        self.model = WideResNet(img_size, depth=depth, k=width)()
        self.model.load_weights(pretrained_model)
        self.model = reduce_keras_model(self.model)
    
    def classify(self, image, faces):
        # bgr to rgb
        image = image[...,::-1]
        images = np.zeros((len(faces), self.img_size, self.img_size, 3))

        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                images[i, :, :, :] = cv2.resize(image[y:y+h, x:x+w, :], (self.img_size, self.img_size))
                #print(images[i, :, :, :].max(), images[i, :, :, :].shape)
                #plt.imshow(images[i, :, :, :].astype(np.uint8))
                #plt.show()
        
        results = self.model.predict(images)
        if not len(results):
            return None, None
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        return predicted_genders[:,0], predicted_ages
        
'''        
class FaceClass:
    
    def __init__(self,img_size=64, depth=16, width=8, path = "optimized8bits.tflite"):
        self.img_size = img_size
        self.interpreter =  tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
    
    def classify(self, image, faces):
        if not len(faces):
            return None, None
        # bgr to rgb
        image = image[...,::-1]
        images = np.zeros((len(faces), self.img_size, self.img_size, 3))

        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                images[i, :, :, :] = cv2.resize(image[y:y+h, x:x+w, :], (self.img_size, self.img_size))
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        print(input_details)
        print(output_details)
        
        print(images.shape)
        self.interpreter.set_tensor(input_details[0]['index'], images.astype(np.float32))
        self.interpreter.invoke()
        
        predicted_genders = self.interpreter.get_tensor(output_details[0]['index'])
        predicted_ages = self.interpreter.get_tensor(output_details[1]['index'])
        
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = predicted_ages.dot(ages).flatten()
        print(predicted_genders[:,0], predicted_ages)
        return predicted_genders[:,0], predicted_ages

class FaceClass:
    
    def __init__(self,img_size=64, depth=16, width=8, path = "optimized8bits.tflite"):
        self.img_size = img_size
        self.graph = tf.Graph()
        new_saver = tf.train.import_meta_graph('graph/saved_checkpoint-0.meta', graph=self.graph)
        self.sess = tf.Session(graph=self.graph)
        new_saver.restore(self.sess, 'graph/saved_checkpoint-0')

    def classify(self, image, faces):
    
        if not len(faces):
            return None, None
        # bgr to rgb
        image = image[...,::-1]
        images = np.zeros((len(faces), self.img_size, self.img_size, 3))

        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                images[i, :, :, :] = cv2.resize(image[y:y+h, x:x+w, :], (self.img_size, self.img_size))
        
        d1,d2 = self.graph.get_tensor_by_name("pred_gender/Softmax:0"),self.graph.get_tensor_by_name("pred_age/Softmax:0")
        #with self.sess:
        predicted_genders, predicted_ages = self.sess.run([d1, d2],feed_dict={'input_1:0':images})
                              
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = predicted_ages.dot(ages).flatten()
        print(predicted_genders[:,0], predicted_ages)
        return predicted_genders[:,0], predicted_ages
'''

