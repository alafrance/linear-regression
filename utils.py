import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd


def get_theta():
    with open('data/model/model.pkl', 'rb') as file:
        return pickle.load(file)


def set_theta(theta0, theta1):
    print(theta0, theta1)
    with open('data/model/model.pkl', 'wb+') as file:
        pickle.dump([theta0, theta1], file)


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return x * std + mean


def get_dataset():
    dataset = pd.read_csv('data/raw/data.csv', sep=',')
    [x, y] = [dataset['km'].to_numpy(), dataset['price'].to_numpy()]

    return dataset, x, y


def get_mean_and_std(x, y):
    mean = {'km': np.mean(x), 'price': np.mean(y)}
    std = {'km': np.std(x), 'price': np.std(y)}
    return mean, std


def normalize_data(x, y):
    [mean, std] = get_mean_and_std(x, y)

    x = normalize(x, mean['km'], std['km'])
    y = normalize(y, mean['price'], std['price'])
    return x, y


def denormalize_theta(theta0_normalized, theta1_normalized, x, y):
    [mean, std] = get_mean_and_std(x, y)
    a = theta0_normalized * std['price'] / std['km']
    b = theta1_normalized * std['price'] + mean['price'] - a * mean['km']
    return a, b


def gradient_descent(theta0, theta1, x, y):
    learning_rate = 0.1
    for i in range(1000):
        y_pred = theta0 * x + theta1
        theta0 -= learning_rate * (1 / len(x)) * np.sum((y_pred - y) * x)
        theta1 -= learning_rate * (1 / len(x)) * np.sum(y_pred - y)
    return theta0, theta1


def see_dataset(theta0, theta1, x, y):
    # Plot the linear regression model
    x_values = np.linspace(min(x), max(x), 100)
    plt.plot(x_values, theta0 * x_values + theta1, label=f'Régression linéaire y = {theta0:.2f}x +  {theta1:.2f}', color='red')

    # Plot the data
    plt.scatter(x, y, label='Données')

    # Plot the legend
    plt.xlabel('X')
    plt.ylabel('F(X)')
    plt.legend()

    # Show the plot
    plt.show()

