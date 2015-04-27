all: TS.c City.c City.h
		gcc -c -Wall TS.c -o TS.o
		gcc -c -Wall City.c -o City.o
		ld -r TS.o City.o -o final.o
		gcc -g -Wall final.o -lm
		rm TS.o City.o final.o