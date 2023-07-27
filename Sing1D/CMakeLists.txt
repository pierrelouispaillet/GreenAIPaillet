
LIBDIRE= -L/usr/X11R6/lib64 -L/home/josserand/lib64
INCL = -I/usr/include -I/usr/X11R6/include -I/home/josserand/include
LIB =-lrfftw -lfftw -lm /home/josserand/simulation/parlib/parlib.a
LIBX=-lXaw  -lXt  -lXmu -lX11
SOURCES=sing1D.c integration.c ini_conditions.c readpar.c outils.c mymalloc.c save.c
OBJECTS=sing1D.o integration.o ini_conditions.o readpar.o outils.o mymalloc.o save.o
CFLAGS=-w -O3 -I. -I/home/josserand/simulation/parlib/include -I/usr/include/malloc -I/usr/X11R6/include -I/home/josserand/include
all: Supersolide

clean:
	rm -f *.o *~

Super1D: super1D.c
	cc -w super1D.c $(INCL) $(LIBDIRE) $(LIBX) $(LIB) -o $@

Sing1D: $(OBJECTS)
	cc $(CFLAGS) $(OBJECTS) $(LIBDIRE) $(LIB) -o $@
	mv -f Sing1D /home/josserand/bin/Sing1D