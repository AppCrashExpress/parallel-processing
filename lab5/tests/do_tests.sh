#!/usr/bin/bash

PROG_NAME="./a.out"
TEST_DIR="./tests"
TESTCASE_PATH="${TEST_DIR}/testcases"
COMPARATOR="${TEST_DIR}/comparator"

for tc in `ls ${TESTCASE_PATH}`; do
    printf "Test "${tc}"... "

    tc_path="${TESTCASE_PATH}/${tc}"
    ${PROG_NAME} < "${tc_path}/in.data" | ${COMPARATOR} "${tc_path}/out.data"

    check=$?
    if [ "$check" -eq 0 ]; then
        printf "OK\n"
    else
        printf "Fail\n"
    fi
done
