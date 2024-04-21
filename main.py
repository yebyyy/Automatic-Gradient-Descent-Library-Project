import math

from datasets import load_dataset

from milligrad import Layer


class Model:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.l1 = Layer(input_dim, hidden_dim, activation="relu")
        self.l2 = Layer(hidden_dim, output_dim, activation="relu")

    def __call__(self, x):
        return self.l2(self.l1(x))


def cross_entropy_loss(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    s = 0

    for p, q in zip(y_true, y_pred):
        s += p * q.log()

    return -s


def mse(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    s = 0
    for y_t, y_p in zip(y_true, y_pred):
        s += (y_t - y_p) ** 2

    s *= 1 / len(y_true)
    return s


def image_to_list(im):
    pixels = list(im.getdata())
    for i in range(len(pixels)):
        pixels[i] /= 255
    return pixels


def make_one_hot(index):
    vector = [0 for _ in range(10)]
    vector[index] = 1
    return vector


def main():
    dataset = load_dataset("mnist").shuffle()
    model = Model(28 * 28, 16, 10)

    for example in dataset["train"]:
        image = image_to_list(example["image"])
        label = make_one_hot(example["label"])
        output = model(image)

        loss = mse(y_true=label, y_pred=output)
        # loss = cross_entropy_loss(y_true=label, y_pred=output)
        print(loss.value, [o.value for o in output])

        loss.backward()
        loss.step(learning_rate=0.001)
        loss.zero_grad()


if __name__ == "__main__":
    main()
