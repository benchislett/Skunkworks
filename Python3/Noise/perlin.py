import numpy as np

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


def perlin_noise(res, boxes):
    w, h = res
    num_boxes_x, num_boxes_y = boxes
    box_x, box_y = w // num_boxes_x, h // num_boxes_y

    angles = 2 * np.pi * np.random.rand(num_boxes_y + 1, num_boxes_x + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    noise = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            i = x // box_x
            j = y // box_y

            u = (x / box_x) % 1
            v = (y / box_y) % 1

            weight_x, weight_y = fade(u), fade(v)

            dotted_vecs = []
            for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                grad = gradients[j + dy, i + dx, :]
                dist = [u - dx, v - dy]
                dotted_vecs.append(grad.dot(dist))

            lerp_top = lerp(dotted_vecs[0], dotted_vecs[2], weight_x)
            lerp_bot = lerp(dotted_vecs[1], dotted_vecs[3], weight_x)

            noise[y, x] = lerp(lerp_top, lerp_bot, weight_y)

    return scale(noise, -1, 1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    res = [256, 256]
    boxes = [4, 4]
    noise = perlin_noise(res, boxes)

    plt.imshow(noise, cmap='gray')
    plt.colorbar()
    plt.show()
