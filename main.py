import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd

default_a = 0
default_b = 0
learning_rate = 0.1
num_iterations = 1000
train = True if len(sys.argv) > 1 and sys.argv[1] == 'train' else False


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return x * std + mean


def get_dataset_and_normalization():
    dataset = pd.read_csv('data.csv', sep=',')
    [km, price] = [dataset['km'].to_numpy(), dataset['price'].to_numpy()]
    mean = {'km': np.mean(km), 'price' : np.mean(price)}
    std = {'km': np.std(km), 'price': np.std(price)}
    x = normalize(km, mean['km'], std['km'])
    y = normalize(price, mean['price'], std['price'])
    return x, y, mean, std


def create_linear_regression_model(x, y):
    a, b = get_coefficient()
    if already_trained(a, b) and not train:
        return a, b
    a, b = gradient_descent(default_a, default_b, x, y)
    set_coefficient(a, b)
    return a, b


def already_trained(a, b):
    return a != default_a or b != default_b


def get_coefficient():
    try:
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return [default_a, default_b]


def set_coefficient(a, b):
    with open('model.pkl', 'wb+') as file:
        pickle.dump([a, b], file)


def gradient_descent(a, b, x, y):
    for i in range(num_iterations):
        y_pred = a * x + b
        a -= learning_rate * (1 / len(x)) * np.sum((y_pred - y) * x)
        b -= learning_rate * (1 / len(x)) * np.sum(y_pred - y)
    return a, b


def see_dataset(a_normalized, b_normalized, x_normalized, y_normalized, mean, std):
    # Plot the linear regression model
    x = denormalize(x_normalized, mean['km'], std['km'])
    y = denormalize(y_normalized, mean['price'], std['price'])
    a = a_normalized * std['price'] / std['km']
    b = b_normalized * std['price'] + mean['price'] - a * mean['km']
    x_values = np.linspace(min(x), max(x), 100)
    plt.plot(x_values, a * x_values + b, label=f'Régression linéaire y = {a:.2f}x +  {b:.2f}', color='red')

    # Plot the data
    plt.scatter(x, y, label='Données')

    # Plot the legend
    plt.xlabel('X')
    plt.ylabel('F(X)')
    plt.legend()

    # Show the plot
    plt.show()


def estimate_price(a, b, mileage):
    return a * mileage + b


def main():
    x_normalized, y_normalized, mean, std = get_dataset_and_normalization()
    a_normalized, b_normalized = create_linear_regression_model(x_normalized, y_normalized)

    see_dataset(a_normalized, b_normalized, x_normalized, y_normalized, mean, std)


main()
