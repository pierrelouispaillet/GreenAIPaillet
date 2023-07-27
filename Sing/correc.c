#include<stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>

extern double atof();

double *spec,*ks;
FILE *fspec,*fexpo;

const double    pi = 3.1415926535897932385;

int compt;


main(argc,argv)
int argc;
char **argv;	 
{
	int ii,jj,tmax,kk,Ns,N,Np,ipeak,N1,N2;
	char nom[256];
	float tt,nt,et,ei,ed;
	double moyi,moyd,moym,dtmoy,tpeak,epeak,max,t1,t2,tome,dt;
	if(argc!=3){
		fprintf(stderr,"Usage: %s N name",argv[0]);
	}
	N=atoi(argv[1]);
	dt=0.01;
	tome=0.0001;
		fspec=fopen("amp1","r");
		fexpo=fopen(argv[2],"w");
		for(jj=0;jj<N;jj++){
			fscanf(fspec,"%F %F %F %F %F\n",&tt,&nt,&et,&ei,&ed);
			fprintf(fexpo,"%8.4f %g %g %g %g\n",tome,nt,et,ei,ed);
			tome+=dt;
}
		fclose(fspec);
		fclose(fexpo);
}
