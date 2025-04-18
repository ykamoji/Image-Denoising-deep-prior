import numpy as np


def dot(x, w):
    return np.dot(x, w)


def cross_entropy_gradients(w, x, y, lamda):
    evals = dot(x, w)
    exp_scores = np.exp(evals - np.max(evals, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    probs[np.arange(np.size(y)), y] -= 1
    probs /= np.size(y)
    dw = dot(x.T, probs)
    dw += lamda * w
    return dw


def train(x, y, p):
    param = {}
    param['lambda'] = p['lambda']  # Regularization term
    param['maxiter'] = p['maxiter']  # Number of iterations
    param['eta'] = p['eta']  # Learning rate
    param['addBias'] = p['addBias']  # Weather to add bias to features
    return multiclassLRTrain(x, y, param)


def predict(model, x):
    x_t = x.T
    if model['params']['addBias']:
        x_t = np.c_[np.ones(x.shape[1]), x.T]
    return dot(x_t, model['w']).argmax(axis=1)


def multiclassLRTrain(x, y, params):
    classLabels = np.unique(y)
    numClass = classLabels.shape[0]
    numFeats = x.shape[0]
    model = {'classLabels': classLabels}

    y = y.reshape(1, y.shape[0])
    x_t = x.T
    if params['addBias']:
        x_t = np.c_[np.ones(x.shape[1]), x.T]
        numFeats += 1

    model['params'] = params

    # Randomly initializes a model
    model['w'] = np.random.randn(numFeats, numClass) * 0.01

    for i in range(params['maxiter']):
        model['w'] -= params['eta'] * cross_entropy_gradients(model['w'], x_t, y, params['lambda'])

        # if (i+1) % 50 == 0:
            # accuracy = np.sum(predict(model, x) == y) / y.shape[1]
            # print(f"Epoch [{i+1}/{params['maxiter']}]: Train Accuracy={accuracy:.3f}")

    # print(f"Final Accuracy {np.sum(predict(model, x) == y) / y.shape[1]:.3f}")

    return model
