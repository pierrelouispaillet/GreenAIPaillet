#include<stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>

extern double atof();

double *spec,*ks;
FILE *fdata,*fout;

const double    pi = 3.1415926535897932385;

int compt;


main(argc,argv)
int argc;
char **argv;	 
{
	int ii,jj,tmax,kk,Ns,N,Np,ipeak,N1,N2;
	char nom[256];
	float tt,nt,et,ei,ed;
	double moyi,moyd,moym,dtmoy,tpeak,epeak,max,t1,t2;


	if(argc!=4){
		fprintf(stderr,"Usage: %s N Nstart file",argv[0]);
	}
	N=atoi(argv[1]);
	Ns=atoi(argv[2]);
	fdata=fopen(argv[3],"r");
	sprintf(nom,"%s-light",argv[3]);
	fout=fopen(nom,"w");
	for(ii=0;ii<N/Ns;ii++){
		for(jj=0;jj<Ns;jj++)
			fscanf(fdata,"%F %F %F %F %F\n",&tt,&nt,&et,&ei,&ed);
	fprintf(fout,"%g %g %g\n",tt,nt,ed);
		}
		fclose(fdata);
		fclose(fout);
	}
