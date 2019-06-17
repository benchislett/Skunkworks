default: main

main: main.cpp
	g++ main.cpp -o main.out -std=c++17 -lsfml-graphics -lsfml-window -lsfml-system -Wno-deprecated-declarations

.PHONY: clean
clean:
	rm -f *.out *.o
