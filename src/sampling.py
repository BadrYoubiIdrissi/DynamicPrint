import numpy as np


def scalar(range):
    return np.random.uniform(range[0], range[1])

def int(range):
    return np.random.randint(range[0], range[1])

def dictionary(value):
    if type(value) is dict:
        return {
            k: dictionary(v) for k,v in value.items()
        }
    else:
        return scalar(value)

def position(pos):
    pos = dictionary(pos)
    return np.array([pos["x"], pos["y"], 0])


def color(color):
    color = dictionary(color)
    return np.array([color["hue"], color["saturation"], color["lightness"]])