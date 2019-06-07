# Video to Ascii

Converts a video into ascii and renders in the terminal using `curses`.
The algorithm used is as follows:

1. Load the specified font
2. Write the character list into images
3. Compute the brightness of each character and sort them accordingly
4. For each block of the size of a character, compute the brightness of the source frame
5. Use the brightness to select the nth character assuming linear distribution (this is key, since it allows different characters to create contrast)
6. Render the scaled image into the terminal

When the program terminates, it saves the text into [an output file](./output/output.txt):
