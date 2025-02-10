import pylab as pl
from IPython import display
import time
import numpy as np
import matplotlib.pyplot as plt

def prediction(c, X):
    probabilities = 1 / (1 + np.exp(-X @ c))
    return np.round(probabilities)

def log_loss(c, X, y):
    predictions = 1 / (1 + np.exp(-X @ c))
    return -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))


def gradient(c, X, y):
    m = X.shape[0]
    predictions = 1 / (1 + np.exp(-X @ c))
    return (1/m) * (X.T @ (predictions - y))


def afficher_model(c, X):
    line = -c[0]/c[2] - c[1]/c[2]*X[:,0]
    plt.plot(X[:,0], line, '--')
    display.clear_output(wait=True)
    display.display(pl.gcf())
    time.sleep(2.0)
    return

def RL_GD(X, y, T, eta, interval=50, affichage=True):
    c = np.random.normal(0, 0.1, X.shape[1])
    if affichage: afficher_model(c, X)

    m = X.shape[0]
    for t in range(T + 1):
        c -= eta * gradient(c, X, y)

        loss = log_loss(c, X, y)
        preds = prediction(c, X)
        acc = np.mean(preds == y)

        if t % interval == 0:
            print(f"Iteration: {t}, Log Loss: {loss:.2f}, Accuracy: {acc:.2f}")
            if affichage: afficher_model(c, X)

    return c
