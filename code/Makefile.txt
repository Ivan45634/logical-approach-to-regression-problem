OBJECTS=main.o
CC=g++-5
INC_PATH="$(shell pwd)/include"
CFLAGS=-Wall -Wextra -Werror -Wpedantic -Wno-unused-parameter -std=c++14 -O2 -DBOOST_LOG_DYN_LINK -I$(INC_PATH) -I/usr/include/
CLIBS= -L/usr/local/lib -lboost_log -lpthread -lboost_thread -lboost_system -lboost_log_setup

all : main

main : $(OBJECTS)
	$(CC) $(CFLAGS) $^ $(CLIBS) -o $@

main.o : main.cpp
	$(CC) $(CFLAGS) -c $< -o $@


clean :
	rm -f *.o *.log main

