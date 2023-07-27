#ifndef UTILF_H
#define UTILF_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw.h>
#include <rfftw.h>
#include <time.h>
#include<gsl/gsl_sf_bessel.h>

#ifndef pi
#define pi 3.14159265359
#endif

#define sq(x)		((x) * (x))
#define cube(x)         ((x) * (x) * (x))
#define MAX(x, y)	((x) > (y) ? (x) : (y))
#define MIN(x, y)	((x) < (y) ? (x) : (y))
#define POWER(x, y)	exp((y)*log(x))

typedef fftw_real freal;
typedef freal **freal2D;
typedef double real;
typedef real *real1D;
typedef real **real2D;
typedef fftw_complex comp;
typedef comp **comp2D;
typedef comp *comp1D;
#endif
