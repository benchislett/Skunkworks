from PIL import Image, ImageFont, ImageDraw
import numpy as np

FONT_FILE = 'ubuntu_terminal.ttf'
FONT_SIZE = 16
BACKGROUND_COLOUR = (255,255,255)
CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%?&*()/\.";:,^'

font = ImageFont.truetype(font=FONT_FILE, size=FONT_SIZE)
font_size = font.getsize()
image_size = (font[0] * len(CHARS), font[1])

output = Image.new('RGB', image_size, BACKGROUND_COLOUR)
draw = ImageDraw.Draw(output)

draw.text((0, 0), CHARS, font=font)

font_pixels = np.asarray(output).astype(np.uint8)

def main():
    output.save(FONT_FILE.split('.')[0] + '.bmp', 'BMP')

if __name__ == '__main__':
    main()