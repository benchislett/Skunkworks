default: main

main: main.cpp
	g++ main.cpp -o main.out -lsfml-graphics -lsfml-window -lsfml-system -Wno-deprecated-declarations -o tetris
# Deprecated declaration warning is only SFML/capture, used to snapshot the screen when the window terminates

.PHONY: clean
clean:
	# Remove output and object file(s)
	rm -f *.out *.o ./tetris
	# Reset output image
	git checkout -- output/output.png
	# Reset highscore
	echo -n 0 > data/highscore.cfg
