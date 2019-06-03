import numpy as np
from PIL import Image, ImageSequence
import curses

from process_font import font_pixels, font_size, CHARS

VIDEO_FILE = 'test.wav'

stdscr = curses.initscr() # initialize the screen
curses.noecho() # hide user key input

vid = Image.open(VIDEO_FILE)

def main():
    for frame in ImageSequence.iterator(vid):
        pass

# execute
if __name__ == '__main__':
    try:
        main()
    except:
        pass

# teardown
curses.echo()
curses.endwin()
