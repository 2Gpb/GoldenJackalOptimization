import math
import numpy as np


def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def u_fun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


def f1(x):
    # s = numpy.sum(x ** 2)
    s = 0
    for i in range(0, len(x)):
        s += x[i] ** 2
    return s


def f2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o


def f3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (np.sum(x[0:i])) ** 2
    return o


def f4(x):
    o = max(abs(x))
    return o


def f5(x):
    dim = len(x)
    o = np.sum(
        100 * (x[1:dim] - (x[0: dim - 1] ** 2)) ** 2 + (x[0: dim - 1] - 1) ** 2
    )
    return o


def f6(x):
    o = np.sum(abs((x + 0.5)) ** 2)
    return o


def f7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = np.sum(w * (x ** 4)) + np.random.uniform(0, 1)
    return o


def f8(x):
    o = sum(-x * (np.sin(np.sqrt(abs(x)))))
    return o


def f9(x):
    dim = len(x)
    o = np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * dim
    return o


def f10(x):
    dim = len(x)
    o = (
            -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
            - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
            + 20
            + np.exp(1)
    )
    return o


def f11(x):
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = np.sum(x ** 2) / 4000 - prod(np.cos(x / np.sqrt(w))) + 1
    return o


def f12(x):
    dim = len(x)
    o = (math.pi / dim) * (
            10 * ((np.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
            + np.sum((((x[: dim - 1] + 1) / 4) ** 2) * (1 + 10 * (np.sin(math.pi * (1 + (x[1:] + 1) / 4))) ** 2))
            + ((x[dim - 1] + 1) / 4) ** 2
    ) + np.sum(u_fun(x, 10, 100, 4))
    return o


def f13(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)

    o = 0.1 * (
            (np.sin(3 * np.pi * x[:, 0])) ** 2
            + np.sum((x[:, :-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[:, 1:])) ** 2), axis=1)
            + ((x[:, -1] - 1) ** 2) *
            (1 + (np.sin(2 * np.pi * x[:, -1])) ** 2)
    ) + np.sum(u_fun(x, 5, 100, 4))
    return o


def get_function_param(a):
    param = {
        "f1": ["f1", -100, 100, 30],
        "f2": ["f2", -10, 10, 10],
        "f3": ["f3", -100, 100, 10],
        "f4": ["f4", -100, 100, 10],
        "f5": ["f5", -30, 30, 10],
        "f6": ["f6", -100, 100, 10],
        "f7": ["f7", -1.28, 1.28, 10],
        "f8": ["f8", -500, 500, 10],
        "f9": ["f9", -5.12, 5.12, 10],
        "f10": ["f10", -32, 32, 10],
        "f11": ["f11", -600, 600, 10],
        "f12": ["f12", -50, 50, 10],
        "f13": ["f13", -50, 50, 10]
    }
    return param.get(a, "nothing")
