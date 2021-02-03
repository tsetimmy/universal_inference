import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
tf.keras.backend.set_floatx('float64')

import sys
import os
if sys.platform == 'darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'#Hacky workaround for an error thrown on macOS.

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

class ANN(Model):
    def __init__(self, X_dim, Y_dim, activation):
        super(ANN, self).__init__()

        if activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        elif activation == 'relu':
            self.activation = tf.nn.relu

        self.mask = tf.compat.v1.get_variable(name='mask', shape=X_dim, dtype=tf.float64)
        self.hidden = layers.Dense(25, activation=self.activation)
        self.out = layers.Dense(Y_dim)

    def call(self, X):
        h = tf.multiply(X, self.mask)
        h = self.hidden(h)
        return self.out(h)

class ANN_wrapper:
    def __init__(self, ann, optimizer):
        self.ann = ann
        self.optimizer = optimizer

    def se(self, y_pred, y_true):
        loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        return tf.reduce_sum(loss)

    def run_optimization(self, x, y):
        with tf.GradientTape() as g:
            pred = self.ann(x)
            loss = self.se(pred, y)

        trainable_variables = self.ann.trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    def get_error(self, x, y):
        pred = self.ann(x)
        loss = self.se(pred, y)
        return loss

def generate_data(n, loc, scale):
    X = np.random.normal(size=[n, 2])
    Y = np.sin(5. * X[:, 0:1]) / (5. * X[:, 0:1])
    epsilon = np.random.normal(loc=loc, scale=scale, size=Y.shape)
    Y += epsilon
    return X, Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_noise', type=float, default=0.)
    parser.add_argument('--scale_noise', type=float, default=.01)
    parser.add_argument('--n_train', type=int, default=1000)

    parser.add_argument('--n_validation', type=int, default=1000)
    #parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=500)
    #parser.add_argument('--n_aux_vars', type=int, default=2)
    parser.add_argument('--activation', type=str, default='sigmoid', choices=['sigmoid', 'relu'])

    #parser.add_argument('--discretization_approach', default=False, action='store_true')
    #parser.add_argument('--limiting_normal_dim', type=int, default=500)
    #parser.add_argument('--n_samples', type=int, default=10000)

    args = parser.parse_args()
    print(sys.argv)
    print(args)

    loc_noise = args.loc_noise
    scale_noise = args.scale_noise
    n_train = args.n_train
    n_validation = args.n_validation
    #n_test = args.n_test
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    activation = args.activation

    X_train, Y_train = generate_data(n_train, loc_noise, scale_noise)
    X_validation, Y_validation = generate_data(n_validation, loc_noise, scale_noise)
    #X_test, Y_test = generate_data(n_test, loc_noise, scale_noise)

    train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_data = train_data.repeat().batch(batch_size)

    max_steps = int(n_train / batch_size) * max_epochs

    ann_wrapper = ANN_wrapper(ann=ANN(X_dim=X_train.shape[-1], Y_dim=Y_train.shape[-1], activation=activation),
                              optimizer=tf.keras.optimizers.Adam())

    X_test = np.repeat(np.expand_dims(np.linspace(-5., 5., 10000), axis=-1), X_train.shape[-1], axis=-1)

    pred_before = ann_wrapper.ann(X_test)



    prev_validation_error = np.inf
    epoch = 0
    early_stopping = np.inf
    with tqdm(total=max_steps) as pbar:
        for step, (batch_X, batch_Y) in enumerate(train_data.take(max_steps), 1):
            loss = ann_wrapper.run_optimization(batch_X, batch_Y)
            pbar.update(1)
            if step % 1000 == 0:
                print(loss)
            if step % int(n_train / batch_size) == 0:
                epoch += 1
                validation_error = ann_wrapper.get_error(X_validation, Y_validation).numpy()
                print('Epoch: %i, step: %i, validation error: %f' % (epoch, step, validation_error))
                early_stopping = early_stopping + 1 if prev_validation_error - validation_error <= 1e-5 else 0
                if early_stopping == 5:
                    print('Validation error did not fall below 1e-5 for 5 consecutive epochs; applying early stopping.')
                    break
                prev_validation_error = validation_error
    #print('Done training.')
    #print('Test error (mean): %f.' % ann_wrapper.get_error(X_test, Y_test).numpy() / float(len(X_test)))
    exit()


    pred_after0 = ann_wrapper.ann(np.stack([X_test[:, 0], np.zeros(X_test.shape[0])], axis=-1))
    plt.figure()
    plt.plot(X_test[:, 0], pred_after0)
    plt.title('first dim')
    plt.grid()

    pred_after1 = ann_wrapper.ann(np.stack([np.zeros(X_test.shape[0]), X_test[:, 1]], axis=-1))
    plt.figure()
    plt.plot(X_test[:, 1], pred_after1)
    plt.title('second dim')
    plt.grid()




    plt.show()
    print(ann_wrapper.ann.mask)
    print(ann_wrapper.ann.hidden)
    print(ann_wrapper.ann.out)



    exit()


    pred_after = ann(X_test)


    plt.figure()
    plt.plot(X_test[:, 0], np.sin(5. * X_test[:, 0]) / (5. * X_test[:, 0]), label='true')
    plt.plot(X_test[:, 0], pred_before, label='before')
    plt.plot(X_test[:, 0], pred_after, label='after')

    plt.title('first dim')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.scatter(X_test[:, 1], np.sin(5. * X_test[:, 0]) / (5. * X_test[:, 0]), label='true')
    plt.scatter(X_test[:, 1], pred_before, label='before')
    plt.scatter(X_test[:, 1], pred_after, label='after')

    plt.title('second dim')
    plt.legend()
    plt.grid()




    plt.show()









if __name__ == '__main__':
    main()


