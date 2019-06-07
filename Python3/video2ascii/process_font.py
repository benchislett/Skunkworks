from PIL import Image, ImageFont, ImageDraw
import numpy as np

FONT_FILE = 'UbuntuMono-R.ttf'
FONT_SIZE = 16
FONT_COLOUR = (255, 255, 255)
BACKGROUND_COLOUR = (48, 10, 36)
# CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%?&*()/\.";:,^'
CHARS = " .:-=+/#%@"

font = ImageFont.truetype(font=FONT_FILE, size=FONT_SIZE)
font_size = font.getsize('A')  # monospace font
image_size = (font_size[0] * len(CHARS), font_size[1])

font_image = Image.new('RGB', image_size, BACKGROUND_COLOUR)
draw = ImageDraw.Draw(font_image)

draw.text((0, 0), CHARS, fill=FONT_COLOUR, font=font)


def main():
    font_image.save(FONT_FILE.split('.')[0] + '.bmp', 'BMP')


if __name__ == '__main__':
    main()
