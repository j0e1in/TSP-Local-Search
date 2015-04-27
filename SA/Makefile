all: SA.c City.c City.h
		gcc -c -Wall SA.c -o SA.o
		gcc -c -Wall City.c -o City.o
		ld -r SA.o City.o -o final.o
		gcc -g -Wall final.o -lm
		rm SA.o City.o final.o