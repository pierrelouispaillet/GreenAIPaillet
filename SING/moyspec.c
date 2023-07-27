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
	int ii,jj,tmax,kk,Ns,N,Nf;
	char nom[256];
	float sk,sp;
	double nu,kc;

	if(argc!=5){
		fprintf(stderr,"Usage: %s N Nstart Nfin nu\n",argv[0]);
	}
	N=atoi(argv[1]);
	Ns=atoi(argv[2]);
	Nf=atoi(argv[3]);
	nu=atof(argv[4]);
	kc=exp(-3.*log(nu)/14.);
	spec=(double *)calloc(N,sizeof(double));
	ks=(double *)calloc(N,sizeof(double));
	for(ii=0;ii<N;ii++)
		spec[ii]=0.;
	for(ii=Ns;ii<=Nf;ii++)
	{
		sprintf(nom,"spectre.%d",ii);
		fspec=fopen(nom,"r");
		for(jj=0;jj<N;jj++){
			fscanf(fspec,"%F %F\n",&sk,&sp);
			ks[jj]=sk;
			spec[jj]+=sp;
		}
		fclose(fspec);
	}
		sprintf(nom,"../specmoy-%g",nu);
		fexpo=fopen(nom,"w");
		for(ii=0;ii<N;ii++)
		fprintf(fexpo,"%g %g\n",ks[ii],spec[ii]/(Nf-Ns+1));
		fclose(fexpo);
		sprintf(nom,"../specres-%g",nu);
		fexpo=fopen(nom,"w");
		for(ii=0;ii<N;ii++)
                fprintf(fexpo,"%g %g\n",ks[ii]/kc,exp(log(kc)/3.)*spec[ii]/(Nf-Ns+1));
		fclose(fexpo);
		
}
