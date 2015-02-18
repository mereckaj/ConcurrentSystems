GCC = gcc
ICC = ~/ICC/bin/icc

OPT = -O3 -fopenmp -std=c99 -lm
CFLAG = -march=corei7 -mtune=corei7 -mfpmath=sse,387

FILES = complex-matmul-harness.c
ORIGIN = original.c

OUT_NAME_GCC = GCC_LAB1
OUT_NAME_ORIGIN = GCC_ORIGIN
OUT_NAME_ICC = ICC_LAB1
OUT_NAME_ASM = GCC_ASM.asm

all: icc gcc
gcc:
	$(GCC) $(FILES) -o $(OUT_NAME_GCC) $(OPT) $(CFLAG)
icc:
	$(ICC) $(FILES) -o $(OUT_NAME_ICC) $(OPT)
clean:
	rm $(OUT_NAME_ICC) $(OUT_NAME_GCC)
rebuild: clean all
origin:
	$(GCC) $(ORIGIN) -o $(OUT_NAME_ORIGIN) $(OPT)
asm:
	$(GCC) $(FILES) -o $(OUT_NAME_ASM) $(OPT) $(CLAG) -masm=intel -S -w
