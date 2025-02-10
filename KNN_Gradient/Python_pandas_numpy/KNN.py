import numpy as np



def KNN_algo(X_train, y_train, x_test, k):
    y_prediction = np.zeros(x_test.shape[0])
    dist = np.linalg.norm(X_train[:, np.newaxis] - x_test, axis=2)
    k_indices = np.argsort(dist, axis=0)[:k]
    k_nearest_labels = y_train[k_indices]
    class_count = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(np.unique(y_train))), axis=0,arr=k_nearest_labels)
    y_prediction = np.argmax(class_count, axis=0)

    return y_prediction


def evaluate_accuracy(y_prediction, y_test):
    correct_predictions = np.sum(y_prediction == y_test)
    accuracy = (correct_predictions / len(y_test)) * 100

    return accuracy




