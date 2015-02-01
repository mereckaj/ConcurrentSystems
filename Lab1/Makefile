GCC = gcc
ICC = ~/ICC/bin/icc

OPT = -O3

FILES = complex-matmul-harness.c

OUT_NAME_GCC = GCC_LAB1
OUT_NAME_ICC = ICC_LAB1

all: icc gcc
gcc:
	$(GCC) $(FILES) -o $(OUT_NAME_GCC) $(OPT)
icc:
	$(ICC) $(FILES) -o $(OUT_NAME_ICC) $(OPT)
clean:
	rm $(OUT_NAME_ICC) $(OUT_NAME_GCC)
rebuild: clean all