import matplotlib.pyplot as plt


def draw(x_axis, y_axis):
    plt.plot(x_axis, y_axis)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss figure')
    plt.savefig('loss.jpg')
    # plt.show()


if __name__ == '__main__':
    x_axis = [1.23, 2, 3, 4, 5, 6]
    y_axis = [1, 2, 3, 4, 5, 6]
    draw(x_axis, y_axis)
