CC=nvcc 
CFLAGS=-ccbin=mpic++ --std=c++11 -Werror cross-execution-space-call -lm
BIN=lab8
SRC=heateq.cu


all: ${BIN}

${BIN}: ${SRC}
	${CC} ${CFLAGS} -o ${BIN} ${SRC}

clean:
	rm -f ${BIN}

