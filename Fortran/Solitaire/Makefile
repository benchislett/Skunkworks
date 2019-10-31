MAINFILE = solitaire.f08
OUTFILE = solitaire

# Fortran Compiler
FC := gfortran

# Compiler Flags
FCFLAGS = -o $(OUTFILE)

$(OUTFILE): $(MAINFILE)
		clear
		$(FC) $(FCFLAGS) $^

.PHONY: clean
clean:
		rm -rf *.o *.out $(OUTFILE)
		clear
