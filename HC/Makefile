all: HC.c City.c City.h
		gcc -c -Wall HC.c -o HC.o
		gcc -c -Wall City.c -o City.o
		ld -r HC.o City.o -o final.o
		gcc -g -Wall final.o -lm
		rm HC.o City.o final.o