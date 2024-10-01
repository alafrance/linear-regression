from utils import get_theta


def predict(theta_0, theta_1, mileage):
    return theta_0 * mileage + theta_1


if __name__ == '__main__':
    try:
        try:
            a, b = get_theta()
        except FileNotFoundError:
            a, b = 0, 0
        mileage = float(input("Please enter a mileage: "))
        print("Your prediction is: ", predict(a, b, mileage))
    except FileNotFoundError as e:
        print(e)
    except ValueError:
        print("Please enter a valid mileage")
