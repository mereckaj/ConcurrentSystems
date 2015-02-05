GCC = gcc
ICC = ~/ICC/bin/icc

OPT = -O0 -fopenmp -std=c99

FILES = complex-matmul-harness.c
ORIGIN = original.c

OUT_NAME_GCC = GCC_LAB1
OUT_NAME_ORIGIN = GCC_ORIGIN
OUT_NAME_ICC = ICC_LAB1

ASSEMBLY = -S -masm=intel

all: icc gcc
gcc:
	$(GCC) $(FILES) -o $(OUT_NAME_GCC) $(OPT)
icc:
	$(ICC) $(FILES) -o $(OUT_NAME_ICC) $(OPT)
clean:
	rm $(OUT_NAME_ICC) $(OUT_NAME_GCC)
rebuild: clean all
asm:
	$(GCC) $(FILES) -o $(OUT_NAME_GCC) $(OPT) $(ASSEMBLY)
	$(ICC) $(FILES) -o $(OUT_NAME_ICC) $(OPT) $(ASSEMBLY)
origin:
	$(GCC) $(ORIGIN) -o $(OUT_NAME_ORIGIN) $(OPT)