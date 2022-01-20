import numpy as np
from sklearn import linear_model
from sklearn.metrics import precision_recall_curve, auc


def generate_D2D_features(source_features, terminal_featues):
    """
    :param source_features: list of experiments each of size [n_samples, n_sources, 2]
    :param terminal_featues: list of experiments each of size [n_samples, n_terminals, 2]
    :return: a numpy arrat of features of size [n_experiments, n_samples)
    """

    features = []
    for exp_idx in range(len(source_features)):
        source_sum = np.sum(source_features[exp_idx], axis=1)
        terminal_sum = np.sum(terminal_featues[exp_idx], axis=1)

        numerator = source_sum[:, 0] * terminal_sum[:, 1]
        denominator = source_sum[:, 1] * terminal_sum[:, 0]
        features.append(numerator/denominator)

    features = np.array(features).T
    return features

def generate_D2D_features_2(source_features, terminal_featues):
    """
    :param source_features: list of experiments each of size [n_samples, n_sources, 2]
    :param terminal_featues: list of experiments each of size [n_samples, n_terminals, 2]
    :return: a numpy arrat of features of size [n_experiments, n_samples)
    """

    features = []
    for exp_idx in range(len(source_features)):
        source_sum = np.sum(source_features[exp_idx], axis=1)
        terminal_sum = np.sum(terminal_featues[exp_idx], axis=1)

        features.append([source_sum[:, 0], terminal_sum[:, 1], source_sum[:, 1], terminal_sum[:, 0]])

    features = np.array(features)
    features = np.array(features).reshape(-1, features.shape[-1])
    features = np.array(features).T

    return features

def eval_D2D(train_features, test_features):

    inverse_train_features = 1 / train_features
    inverse_test_features = 1/test_features

    train_labels, test_labels = np.zeros(train_features.shape[0] * 2),  np.zeros(test_features.shape[0] * 2)
    train_labels[:train_features.shape[0]] = 1
    test_labels[:test_features.shape[0]] = 1

    train_features = np.vstack([train_features, inverse_train_features])
    test_features = np.vstack([test_features, inverse_test_features])

    quantized_train_features = np.argsort(np.argsort(train_features, axis=1), axis=1)
    quantized_test_features = np.argsort(np.argsort(test_features, axis=1), axis=1)

    clf = linear_model.LogisticRegression(solver='liblinear')
    probs = clf.fit(quantized_train_features, train_labels).predict_proba(quantized_test_features)
    acc = np.mean(np.argmax(probs, 1) == test_labels)
    precision, recall, thresholds = precision_recall_curve(test_labels, probs[:,1])
    mean_auc = auc(recall, precision)
    return probs, precision, recall, mean_auc

def eval_D2D_2(train_features, test_features):

    inverse_idx = np.ravel([np.array([2, 3, 0, 1]) + 4*i  for i in range(int(train_features.shape[1]//4))])
    inverse_train_features = train_features[: , inverse_idx]
    inverse_test_features = test_features[: , inverse_idx]

    train_labels, test_labels = np.zeros(train_features.shape[0] * 2), np.zeros(test_features.shape[0] * 2)
    train_labels[:train_features.shape[0]] = 1
    test_labels[:test_features.shape[0]] = 1

    train_features = np.vstack([train_features, inverse_train_features])
    test_features = np.vstack([test_features, inverse_test_features])
    #
    quantized_train_features = np.argsort(np.argsort(train_features, axis=1), axis=1)
    quantized_test_features = np.argsort(np.argsort(test_features, axis=1), axis=1)

    clf = linear_model.LogisticRegression(solver='liblinear')
    probs = clf.fit(quantized_train_features, train_labels).predict_proba(quantized_test_features)
    acc = np.mean(np.argmax(probs, 1) == test_labels)
    precision, recall, thresholds = precision_recall_curve(test_labels, probs[:, 1])
    mean_auc = auc(recall, precision)
    return probs, precision, recall, mean_auc