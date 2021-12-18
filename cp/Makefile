CC:=g++
CFLAGS:=-O3
TARGET:=renderer

SRC_DIR:=./src
OBJ_DIR:=./obj

SRCS := $(wildcard ${SRC_DIR}/*.cpp)
OBJS := $(patsubst ${SRC_DIR}/%.cpp, ${OBJ_DIR}/%.o, ${SRCS})


${TARGET}: ${OBJS}
	g++ -o $@ $^

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp
	@mkdir -p $(@D)
	${CC} ${CFLAGS} -c -o $@ $<

clean:
	rm -r ${OBJ_DIR}
	rm -f ${TARGET}
