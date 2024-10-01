from utils import get_dataset, get_theta, normalize_data
from predict import predict


def calculate_r_squared():
    dataset, x_values, y_values = get_dataset()
    theta_0, theta_1 = get_theta()

    y_predicted = [predict(theta_0, theta_1, x) for x in x_values]
    y_mean = sum(y_values) / len(y_values)
    ss_res = sum((y_obs - y_pred) ** 2 for y_obs, y_pred in zip(y_values, y_predicted))
    ss_tot = sum((y_obs - y_mean) ** 2 for y_obs in y_values)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


if __name__ == '__main__':
    try:
        r2 = calculate_r_squared()
        print(f"Le coefficient de détermination R² est : {r2:.4f}")
    except FileNotFoundError as e:
        print("Tu n'as pas encore entraîné ton modèle !")
