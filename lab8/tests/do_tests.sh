#!/usr/bin/bash

PROG_NAME="./lab8"
TEST_DIR="./tests"
COMPARATOR="${TEST_DIR}/comparator"
TEST_FILE="in.txt"

mpirun -np $((`head -n 1 ${TEST_FILE} | sed 's/ /*/g'`)) ${PROG_NAME} < "${TEST_FILE}"
python "${TEST_DIR}/solve.py" < "${TEST_FILE}" | ${COMPARATOR} "mpi.out"

check=$?
if [ "$check" -eq 0 ]; then
    printf "OK\n"
else
    printf "Fail\n"
fi
