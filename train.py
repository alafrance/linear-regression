from utils import gradient_descent, get_dataset, normalize_data, see_dataset, set_theta, denormalize_theta


def train(x, y):
    return gradient_descent(0, 0, x, y)


def main():
    dataset, x, y = get_dataset()
    x_normalized, y_normalized = normalize_data(x, y)
    theta0_normalized, theta1_normalized = train(x_normalized, y_normalized)
    theta0, theta1 = denormalize_theta(theta0_normalized, theta1_normalized, x, y)

    set_theta(theta0, theta1)
    see_dataset(theta0, theta1, x, y)


if __name__ == '__main__':
    main()

