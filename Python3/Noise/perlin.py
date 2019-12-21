import numpy as np
import itertools
from operator import sub

def fade(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def lerp(a, b, w):
    return (1 - w) * a + w * b


def scale(x, new_min, new_max):
    x -= x.min()
    x /= x.max()
    x *= new_max - new_min
    x += new_min
    return x


def scale_ndarray(x, new_shape):
    factors = tuple(i // j for (i, j) in zip(new_shape, x.shape))
    y = np.zeros(new_shape)
    
    for slice_starts in itertools.product(*(range(i) for i in factors)):
        slices = tuple(slice(i, j, k) for (i, j, k) in zip(slice_starts, new_shape, factors))
        y[slices] = x

    return y


def merge_vecs(vecs, weights):
    acc = vecs.copy()
    r = 0
    while len(acc) > 1:
        for i in range(len(acc) // 2):
            w = weights[r]
            acc[i] = lerp(acc[i], acc[len(acc) // 2 + i], w)
        acc = acc[:len(acc) // 2]
        r += 1
    return acc[0]


def perlin_noise(res, boxes):
    n = len(res)
    dims = np.asarray(res)
    boxes = np.asarray(boxes)
    box_sizes = dims // boxes

    grad_dims = tuple(boxes + 1) + (n,)

    gradients = np.random.normal(size=grad_dims)
    gradients /= np.linalg.norm(gradients, axis=-1, keepdims=True)

    for i in range(n):
        in_idxs = [slice(None) for _ in range(n)]
        out_idxs = in_idxs.copy()
        in_idxs[i] = -1
        out_idxs[i] = 0
        gradients[tuple(in_idxs)] = gradients[tuple(out_idxs)]

    noise = np.zeros(res)

    indexes = tuple(slice(0, i) for i in dims)
    grid = np.moveaxis(np.mgrid[indexes], 0, -1)
    grid_boxes = grid // box_sizes
    grid_fracs = (grid / box_sizes) % 1
    
    weights = np.moveaxis(fade(grid_fracs), -1, 0)

    dotted_vecs = []
    for diff in itertools.product((0, 1), repeat=n):
        indexes = tuple(slice(i, j) for (i, j) in zip(diff, tuple(map(lambda x, y: x - (1 - y), gradients.shape, diff))))
        grads = scale_ndarray(gradients[indexes], grid.shape)
        dists = grid_fracs - diff
        dotted_vecs.append(np.einsum('ijk,ijk->ij', grads, dists))

    noise = merge_vecs(dotted_vecs, weights)

    return scale(noise, -1, 1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    res = [256, 256]
    boxes = [4, 4]
    noise = perlin_noise(res, boxes)
    noise = np.hstack((noise, noise))
    noise = np.vstack((noise, noise))

    plt.imshow(noise, cmap='gray')
    plt.colorbar()
    plt.show()
