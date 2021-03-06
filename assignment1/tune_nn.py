import itertools
import datetime
import os
import multiprocessing
import multiprocessing.queues


import numpy as np


from cs294_129.data_utils import load_CIFAR10


def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[xrange(y.shape[0]), y] = 1
    return encoded

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs294_129/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, one_hot_encode(y_train), X_val, one_hot_encode(y_val), X_test, one_hot_encode(y_test)


def evaluate_nn(X_train, y_train, X_val, y_val, n_epoch=20,
                tune_params=dict(
                    hidden_size=50, batch_size=200,
                    reg=0.5, learning_rate=1e-4,
                    learning_rate_decay=0.95)
                ):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import tensorflow as tf
    import tflearn
    # tf.reset_default_graph()
    tflearn.init_graph()
        # sess = tf.Session()
    net = tflearn.input_data(shape=[None, 3072])
    net = tflearn.fully_connected(
        net, tune_params['hidden_size'],
        activation='relu', weight_decay=tune_params['reg'],
        weights_init=tflearn.initializations.truncated_normal(stddev=1e-4),
    )
    net = tflearn.fully_connected(
        net, 10, activation='softmax', weight_decay=tune_params['reg'],
        weights_init=tflearn.initializations.truncated_normal(stddev=1e-4),
    )
    # optimizer = tflearn.optimizers.SGD(
    #    learning_rate=tune_params['learning_rate'],
    #    lr_decay=tune_params['learning_rate_decay'],
    #    decay_step=max(1, int(X_train.shape[0] / tune_params['batch_size'])),
    #    staircase=True
    # )
    net = tflearn.regression(
        net, optimizer='adam', loss='categorical_crossentropy',
        batch_size=tune_params['batch_size']
    )
    model = tflearn.DNN(net)
    model.fit(
        X_train, y_train, n_epoch=n_epoch,
        batch_size=tune_params['batch_size'], snapshot_epoch=False,
        show_metric=False,
    )
    accuracy = model.evaluate(X_val, y_val)[0]
        # sess.close()
    return accuracy


def evaluate_nn_process(result_queue, *args, **kwargs):
    accuracy = evaluate_nn(*args, **kwargs)
    result_queue.put(accuracy)


def evaluate_nn_wrapped(*args, **kwargs):
    result_queue = multiprocessing.queues.SimpleQueue()
    process = multiprocessing.Process(
        target=evaluate_nn_process,
        args=tuple([result_queue] + list(args)),
        kwargs=kwargs
    )
    process.start()
    process.join()
    return result_queue.get()



if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

    hyperparams = dict(
        # hidden_size=[50, 80, 100, 150, 200, 300, 600, 1000],
        hidden_size=[60, 80, 100, 120],
        # batch_size=[256, 512],
        batch_size=[512],
        reg=(10 ** np.linspace(-3, 3, 16)).tolist(),
        # reg=[0.0464158883361],
        # learning_rate=(10 ** np.linspace(-3, -1.5, 10)).tolist(),
        learning_rate=[0.0001],
        # learning_rate_decay=np.linspace(0.9, 1, 20).tolist(),
        learning_rate_decay=[0.99],
    )
    keys = hyperparams.keys()
    hyperparam_list = [hyperparams[key] for key in keys]

    best_config = None
    best_accuracy = -1

    with open(datetime.datetime.today().strftime("tune_nn_results/%Y-%m-%d_%H-%M.txt"), 'w', 0) as flog:
        flog.write('============= {} =============\n'.format(datetime.datetime.now()))

        all_configs = list(itertools.product(*hyperparam_list))

        for config_idx in xrange(len(all_configs)):


            config = all_configs[config_idx]
            flog.write('    ')
            param = {}
            for i in xrange(len(keys)):
                param[keys[i]] = config[i]
                flog.write('{}: {},  '.format(keys[i], config[i]))

            print 'Training:  {}/{}'.format(config_idx, len(all_configs))
            print param
            accuracy = evaluate_nn_wrapped(X_train, y_train, X_val, y_val, tune_params=param)
            flog.write('accuracy: {}\n'.format(accuracy))

            print 'accuracy: {}'.format(accuracy)

            if accuracy > best_accuracy:
                best_config = config
                best_accuracy = accuracy

            flog.write('    Current best: ')

            for i in xrange(len(keys)):
                flog.write('{}: {},  '.format(keys[i], best_config[i]))
            flog.write('accuracy: {}\n\n'.format(best_accuracy))

        flog.write('\n    Now showing the best:\n')
        flog.write('    ')

        for i in xrange(len(keys)):
            flog.write('{}: {},  '.format(keys[i], best_config[i]))

        flog.write('accuracy: {}\n'.format(best_accuracy))

        flog.write('\n============= END {} =============\n\n\n\n'.format(datetime.datetime.now()))
