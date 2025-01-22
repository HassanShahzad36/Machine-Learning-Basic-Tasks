import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    return sigmoid(linear_model)

def compute_loss(y_true, y_pred):
    n = len(y_true)
    loss = -1 / n * (np.dot(y_true, np.log(y_pred)) + np.dot(1 - y_true, np.log(1 - y_pred)))
    return loss


def gradient_descent(X, y, weights, bias, learning_rate, epochs):
    n = len(y)
    for i in range(epochs):
        y_pred = predict(X, weights, bias)
        dw = 1 / n * np.dot(X.T, (y_pred - y))
        db = 1 / n * np.sum(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
z = np.array([0, 1, 1, 1])

weights = np.zeros(X.shape[1])
bias = 0


learning_rate = 0.1
iterations = 1000

#AND
weights, bias = gradient_descent(X, y, weights, bias, learning_rate, iterations)

predictions = predict(X, weights, bias)
predicted_classes = [1 if p >= 0.5 else 0 for p in predictions]
print("Predicted outputs AND:", predicted_classes)
print("Actual outputs AND:   ", y)
print("Model weights AND:", weights)
print("Model bias AND:", bias)


#OR
weightsOR, biasOR = gradient_descent(X, z, weights, bias, learning_rate, iterations)

predictionsOR = predict(X, weights, bias)
predicted_classesOR = [1 if p >= 0.5 else 0 for p in predictionsOR]


print("Predicted outputs OR:", predicted_classesOR)
print("Actual outputs OR:   ", z)
print("Model weights OR:", weightsOR)
print("Model bias OR:", biasOR)

