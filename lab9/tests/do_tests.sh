#!/usr/bin/bash

PROG_NAME="./lab9"
TEST_DIR="./tests"
COMPARATOR="${TEST_DIR}/comparator"
TEST_FILE="in.txt"

mpirun --oversubscribe -np $((`head -n 1 ${TEST_FILE} | sed 's/ /*/g'`)) "${TEST_DIR}/solve" < "${TEST_FILE}"
mv "mpi.out" "$TEST_DIR"
mpirun --oversubscribe -np $((`head -n 1 ${TEST_FILE} | sed 's/ /*/g'`)) ${PROG_NAME} < "${TEST_FILE}"
cat "${TEST_DIR}/mpi.out" | ${COMPARATOR} "mpi.out"

check=$?
if [ "$check" -eq 0 ]; then
    printf "OK\n"
else
    printf "Fail\n"
fi
