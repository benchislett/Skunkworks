default: main

main: main.c
	gcc main.c -o main.out -I /usr/include/cairo -L /usr/lib -lcairo

.PHONY: clean
clean:
	rm -f *.o *.out