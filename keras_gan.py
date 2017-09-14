from tensorflow.contrib.keras.python.keras.datasets import mnist
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras.python.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Activation, UpSampling2D
from tensorflow.contrib.keras.python.keras.layers.core import Dropout, Dense, Flatten, Reshape
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pdb
from progress_bar import InitBar

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NUM_CLASSES = 10
cwd = os.getcwd()
datapath = os.path.join(cwd, '..', 'jointtensorflowkeras')

def load_dataset():
    # load training data
    training_images = []
    training_labels = []
    print('Loading training data:')
    bar = InitBar()
    for i in range(NUM_CLASSES):
        bar(100.0*float(i)/float(NUM_CLASSES))
        pathname = os.path.join(datapath, 'mnist_png', 'training', str(i))
        for fname in os.listdir(pathname):
            fullfname = os.path.join(pathname, fname)
            img = cv2.imread(fullfname, 0)
            training_images.append(img)
            training_labels.append(i)
    training_images = np.stack(training_images, axis=0)
    training_labels = np.array(training_labels)
    print('training data has the shape of {}\n'.format(training_images.shape))

    # load test data
    test_images = []
    test_labels = []
    print('Loading test data:')
    for i in range(NUM_CLASSES):
        bar(100.0*float(i)/float(NUM_CLASSES))
        pathname = os.path.join(datapath, 'mnist_png', 'testing', str(i))
        for fname in os.listdir(pathname):
            img = cv2.imread(os.path.join(pathname, fname), 0)
            test_images.append(img)
            test_labels.append(i)
    test_images = np.stack(test_images, axis=0)
    test_labels = np.array(test_labels)
    print('test data has the shape of {}\n'.format(test_images.shape))

    return (training_images, training_labels, test_images, test_labels)


def get_real_mbatch(data, batch_sz):
    perm = np.random.choice(data.shape[0], size=batch_sz)
    batch_data = data[perm]
    batch_data = np.expand_dims(batch_data, axis=-1)
    return data[perm]

def get_batch(step, bs, images, labels):                                   
    n = images.shape[0]
    #pdb.set_trace()
    if (step+1)*bs > n:
        return images[step*bs:, :, :, :], labels[step*bs:]
    else:
        return images[step*bs:(step+1)*bs, :, :, :], labels[step*bs:(step+1)*bs]


def main():
    training_images, training_labels, test_images, test_labels = load_dataset()
    
    # plt.imshow(training_images[:,:,0], cmap='gray')
    # plt.show()

    perm_train = np.random.permutation(training_labels.size)
    training_labels = training_labels[perm_train]
    training_images = (training_images[perm_train, :, :] - 127.5) / 127.5
    training_images = np.expand_dims(training_images, -1)
    print(training_images.shape)
    test_images = test_images / 255.0
    test_images = np.expand_dims(test_images, -1)

    # pdb.set_trace()

#    training_labels = to_categorical(training_labels, NUM_CLASSES)
#    test_labels = to_categorical(test_labels, NUM_CLASSES)

    BATCH_SIZE = 32*8
    WIDTH, HEIGHT = 28, 28
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    #####################################
    ### Defiining the Discriminator:
    #####################################
    input_D = Input(shape=(HEIGHT,WIDTH,1), name='input_D')
    x = Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', name='conv1_D')(input_D)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', name='conv2_D')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='dense1_D')(x)
    output_D = Dense(1, activation='sigmoid', name='output_D')(x)
    model_D = Model(inputs=input_D, outputs=output_D)
    model_D.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer(learning_rate=adam_lr, beta1=adam_beta_1), metrics=['accuracy'])

    #####################################
    ### Defiining the Generator:
    #####################################
    LATENT_SIZE = 100
    input_G = Input(shape=(LATENT_SIZE,), name='input_gen')
    x = Dense(7*7*32, activation='linear', name='Dense1_G')(input_G)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((7, 7, 32))(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same', name='conv1_gen')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same', name='conv2_gen')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=1 , kernel_size=1, strides=(1,1), padding='same', name='conv3_gen')(x)
    img_G = Activation('tanh')(x)
    model_G = Model(inputs=input_G, outputs=img_G)
    model_G.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer(learning_rate=adam_lr, beta1=adam_beta_1))

    #####################################
    ### Defiining the Combined GAN:
    #####################################
    model_D.trainable = False # Since model_D is already compiled, thediscriminator model remains trainble, 
    # but here in the combined model it becomes non-trainable
    input_main = Input(shape=(LATENT_SIZE,), name='input_main') # Note that this input should be different from the input to Generator
    combined = Model(inputs=input_main, outputs=model_D(model_G(input_main)))
    combined.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer(learning_rate=adam_lr, beta1=adam_beta_1), metrics=['accuracy'])

    print(combined.summary())

    #####################################
    ### Training:
    #####################################
    bar = InitBar()
    N = training_images.shape[0]
    for iter in range(100):
        fake_input = np.random.randn(1, LATENT_SIZE)
        fake_image = model_G.predict(fake_input)
        loss_G, acc_G, loss_D, acc_D = 0, 0, 0, 0
        steps = (int)(np.ceil(float(N)/float(BATCH_SIZE)))
        for batch_iter in range(steps):
            bar(100.0*batch_iter/float(steps))
            real_image, _ = get_batch(batch_iter, BATCH_SIZE/2, training_images, training_labels)
            ####################
            ## Discriminator Training
            ####################
            #  Note that if using BN layer in Discriminator, minibatch should contain only real images or fake images.
            fake_input = np.random.randn(BATCH_SIZE/2, LATENT_SIZE)
            fake_image = model_G.predict(fake_input)
            #real_image = get_real_mbatch(batch_sz=BATCH_SIZE/2, data=training_images)
            agg_input  = np.concatenate((fake_image, real_image), axis=0)
            agg_output = np.zeros((BATCH_SIZE,))
            agg_output[BATCH_SIZE/2:] = 1
            perm = np.random.permutation(BATCH_SIZE)
            agg_input  = agg_input[perm]
            agg_output = agg_output[perm]
            #pdb.set_trace()
            tr = model_D.train_on_batch(x=agg_input, y=agg_output)
            loss_D += tr[0]
            acc_D  += tr[1]
            #####################
            ## Generator Training
            #####################
            fake_input = np.random.randn(BATCH_SIZE, LATENT_SIZE)
            fake_label = np.ones(BATCH_SIZE,)
            tr = combined.train_on_batch(x=fake_input, y=fake_label)
            loss_G += tr[0]
            acc_G  += tr[1]
        print('\nG_loss = {}, G_acc = {}\nD_loss = {}, D_acc = {}'.format(loss_G/float(steps), acc_G/float(steps), loss_D/float(steps), acc_D/float(steps)))

    for iter in range(10):
        fake_input = np.random.randn(1, LATENT_SIZE)
        fake_image = model_G.predict(fake_input)
        plt.imshow(fake_image[0,:,:,0])
        plt.show()

if __name__ == '__main__':
    main()




