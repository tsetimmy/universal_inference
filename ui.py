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
import uuid
import pickle

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
        return tf.reduce_mean(loss)

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

    assert (args.n_train % 2 == 0), 'provide an even number of training instances.'

    loc_noise = args.loc_noise
    scale_noise = args.scale_noise
    n_train = args.n_train
    n_validation = args.n_validation
    #n_test = args.n_test
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    activation = args.activation

    half = n_train // 2

    X_train_D0, Y_train_D0 = generate_data(half, loc_noise, scale_noise)
    X_validation, Y_validation = generate_data(n_validation, loc_noise, scale_noise)
    #X_test, Y_test = generate_data(n_test, loc_noise, scale_noise)

    train_data = tf.data.Dataset.from_tensor_slices((X_train_D0, Y_train_D0))
    train_data = train_data.repeat().batch(batch_size)

    max_steps = int(half / batch_size) * max_epochs

    ann_wrapper = ANN_wrapper(ann=ANN(X_dim=X_train_D0.shape[-1], Y_dim=Y_train_D0.shape[-1], activation=activation),
                              optimizer=tf.keras.optimizers.Adam())

    X_test = np.repeat(np.expand_dims(np.linspace(-5., 5., 10000), axis=-1), X_train_D0.shape[-1], axis=-1)

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
            if step % int(half / batch_size) == 0:
                epoch += 1
                validation_error = ann_wrapper.get_error(X_validation, Y_validation).numpy()
                print('Epoch: %i, step: %i, validation error: %f' % (epoch, step, validation_error))
                early_stopping = early_stopping + 1 if prev_validation_error - validation_error <= 1e-5 else 0
                if early_stopping == 20:
                    print('Validation error did not fall below 1e-5 for 20 consecutive epochs; applying early stopping.')
                    break
                prev_validation_error = validation_error

    # UI
    alpha = .05
    X_train_D1, Y_train_D1 = generate_data(half, loc_noise, scale_noise)

    pred = ann_wrapper.ann(X_train_D1).numpy()
    denom = np.exp(-.5 * np.linalg.norm(Y_train_D1 - pred)**2.)

    # ----
    original_mask = ann_wrapper.ann.mask.numpy()

    Values = [np.linspace(-10., 10., 20001), np.linspace(-10., 10., 20001) + 1.]
    Values = np.stack(Values, axis=0)
    rejections = np.zeros_like(Values)

    for i in tqdm(range(X_train_D0.shape[-1])):
        values = Values[i]
        mask = original_mask.copy()
        for j in tqdm(range(len(values))):
            mask[i] = values[j]
            ann_wrapper.ann.mask.assign(mask)

            pred = ann_wrapper.ann(X_train_D1).numpy()
            numer = np.exp(-.5 * np.linalg.norm(Y_train_D1 - pred)**2.)

            T = numer / denom

            rejections[i, j] = 1. if T > 1. / alpha else 0.

    filename = str(uuid.uuid4()) + '.pickle'
    print(filename)

    d = {'Values': Values, 'rejections': rejections}
    pickle.dump(d, open(filename, 'wb'))

if __name__ == '__main__':
    main()


