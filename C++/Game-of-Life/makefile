default: main

main: main.cpp
	g++ main.cpp -lsfml-graphics -lsfml-window -lsfml-system -Wno-deprecated-declarations -o gameoflife
# Deprecated declaration warning is only SFML/capture, used to snapshot the screen when the window terminates

.PHONY: clean
clean:
	# Remove any built object files and the main executable
	rm -f *.out *.o gameoflife
	# Reset the output image
	git checkout -- output/output.png
