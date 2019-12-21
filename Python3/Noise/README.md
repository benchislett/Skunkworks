# Random Noise

## Description

Implementation of various noise generation algorithms in Python+Numpy, rendered with Pillow and Matplotlib.

Supported Algorithms:
- Perlin Noise
- Fractal Perlin Noise

Renderings can be done in arbitrary dimensions, though perlin complexity scales exponentially.<br/>
Gradients repeat on all edges, so images are tileable and animations are perfect loops.


## Usage

Call the noise file with python with the `-h` flag to see the following output:

```
usage: perlin.py [-h] [--res RES] [--grid GRID] [--frames FRAMES] [--plot]
                 [--render] [--fractal FRACTAL]

View or animate a perlin noise image.

optional arguments:
  -h, --help         show this help message and exit
  --res RES          Image resolution
  --grid GRID        Grid resolution. Should be a factor of res.
  --frames FRAMES    Number of frames to use when animating.
  --plot             Plot the first frame with matplotlib at runtime.
  --render           Render a gif animation over the noise.
  --fractal FRACTAL  Use fractal noise with specified number of octaves.
```

## Output

### Perlin Noise, RES 256, GRID 8, FRAMES 64 (2.7s)

![Perlin Noise with resolution 256x256 and grid size 8x8, over 64 frames](./output/perlin-256-8.gif)

### Fractal Perlin Noise, RES 256, GRID 8, FRAMES 64, FRACTAL 5 (9.4s)

![Fractal Noise with resolution 256x256 and grid size 8x8, over 64 frames](./output/fractal-256-8.gif)
