import pickle
import random

from keras.datasets import mnist
from keras.models import model_from_json, Model
from sklearn.decomposition import PCA, IncrementalPCA


def main(mode='pca'):
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_train /= 255

    ids = random.sample(range(0, 60000), 10000)
    with open('../models/ids.pkl') as f:
        pickle.dump(ids, f)
    X_train = X_train[ids]
    with open('../models/X_train_10000.pkl') as f:
        pickle.dump(X_train, f)
    y_train = y_train[ids]
    with open('../oss_data/MNIST_labels.tsv') as f:
        for label in y_train:
            f.write(str(label) + '\n')

    with open('../models/config.json') as f:
        config = f.read()
    model = model_from_json(config)
    model.load_weights('../models/best_weight.h5')
    new_model = Model(model.inputs, model.layers[-3].output)
    new_model.set_weights(model.get_weights())

    embs_4096 = new_model.predict(X_train)
    if mode == 'pca':
        pca = PCA(n_components=128)
        embs_128 = pca.fit_transform(embs_4096)
        with open('../models/embs_128D.pkl', 'wb') as f:
            pickle.dump(embs_128, f)
        embs_128.tofile('../oss_data/MNIST_tensor.bytes')
    elif mode == 'ipca':
        # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA  # noqa
        ipca = IncrementalPCA(n_components=128, batch_size=200)
        embs_128 = ipca.fit_transform(embs_4096)
        with open('../models/embs_128D_2.pkl', 'wb') as f:
            pickle.dump(embs_128, f)
        embs_128.tofile('../oss_data/MNIST_tensor_2.bytes')
    else:
        raise NotImplementedError("Mode must be set")


if __name__ == '__main__':
    main()
