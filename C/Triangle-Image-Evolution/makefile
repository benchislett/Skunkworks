default: main

main: main.c
	gcc main.c -I /usr/include/cairo -L /usr/lib -lcairo -o generate

.PHONY: clean
clean:
	# Remove object and executable files
	rm -f *.o *.out generate
	# Reset output image
	git checkout -- output/output.png
