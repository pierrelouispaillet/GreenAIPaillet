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
	double moyi,moyd,moym,dtmoy,tpeak,epeak,max,t1,t2;
	if(argc!=4){
		fprintf(stderr,"Usage: %s N Nstart epeak",argv[0]);
	}
	N=atoi(argv[1]);
	Ns=atoi(argv[2]);
	epeak=atof(argv[3]);
	spec=(double *)calloc(N,sizeof(double));
	ks=(double *)calloc(N,sizeof(double));
	for(ii=0;ii<N;ii++)
		spec[ii]=0.;
		Np=0;
		ipeak=0;
		moyi=0.;
		moyd=0.;
		moym=0.;
		dtmoy=0.;
		ipeak=0;	
		fspec=fopen("amp1","r");
		for(jj=0;jj<Ns;jj++)
			fscanf(fspec,"%F %F %F %F %F\n",&tt,&nt,&et,&ei,&ed);
		N2=jj;	
		for(jj=Ns;jj<=N;jj++){
                        fscanf(fspec,"%F %F %F %F %F\n",&tt,&nt,&et,&ei,&ed);
			moyi+=ei;
			moyd+=ed;
			if(ed>epeak){
				if(ipeak==0){
					ipeak=1;
					Np++;
					N1=jj;
					max=ed;
					dtmoy+=0.01*(N1-N2);
				}
				if(ipeak==1){
					if(ed>max)
						max=ed;
				}
			}
			if((ed<epeak)&(ipeak==1)){
				ipeak=0;
				N2=jj;
				tpeak+=0.01*(N2-N1);
				moym+=max;
			}			
		}
		fclose(fspec);
        printf("Np=%d\n",Np);
	printf("injection=%g, dissipation=%g, diss-max=%g, tpeak=%g, dtpeak=%g\n",moyi/(N-Ns+1.),moyd/(N-Ns+1.),moym/Np,tpeak/Np,dtmoy/Np);
	}
