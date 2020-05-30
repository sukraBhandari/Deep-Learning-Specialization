import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(dim):
    w = np.random.randn(1, dim) * 0.01
    b = 0
    return w, b


def forward_prop(X, w, b):
    Z = np.dot(w, X) + b
    return Z


def cost_function(Z, Y):
    m = Y.shape[1]
    J = (1 / (2 * m)) * np.sum(np.square(Z - Y))
    return J


def back_prop(X, Y, Z):
    m = Y.shape[1]
    dz = (1 / m) * (Z - Y)
    dw = np.dot(dz, X.T)
    db = np.sum(dz)
    return dw, db


def gradient_descent_update(w, b, dw, db, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b


def linear_regression_model(X_train, y_train, X_val, y_val, learning_rate, epochs):
    lenw = X_train.shape[0]
    w, b = initialize_parameters(lenw)

    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]
    for i in range(1, epochs + 1):
        z_train = forward_prop(X_train, w, b)
        cost_train = cost_function(z_train, y_train)
        dw, db = back_prop(X_train, y_train, z_train)
        w, b = gradient_descent_update(w, b, dw, db, learning_rate)

        if i % 100 == 0:
            costs_train.append(cost_train)

            MAE_train = (1 / m_train) * np.sum(np.abs(z_train - y_train))

            z_val = forward_prop(X_val, w, b)
            cost_val = cost_function(z_val, y_val)
            MAE_val = (1 / m_val) * np.sum(np.abs(z_val - y_val))

            print("epochs " + str(i) + '/' + str(epochs) + ': ')
            print('Training cost' + str(cost_train) + '| ' +
                  'validation cost' + str(cost_val))
            print('Training mae' + str(MAE_train) + '| ' +
                  'validation mae' + str(MAE_val))

    plt.plot(costs_train)
    plt.xlabel('Iterations(per 100)')
    plt.ylabel('Training cost')
    plt.title('Learning rate' + str(learning_rate))
    plt.show()
