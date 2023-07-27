#include "utilf.h"
#include "ini_conditions.h"
#include "global.h"
#include "mymalloc.h"
#include "outils.h"

/*------------------------------------------------------------------*/
/*                 Computational arrays                             */
/*------------------------------------------------------------------*/

comp1D psi,psik;
real1D ks;
real1D kx,xm;
real2D fstruct,fstruct2;
int nsauve,N,tmax,tprint,tmax,compt,tdiss,tdi;
double dt,dx,heure,bruit,inv,amp,k0,nu,tome,flux,dissip;
double Mtot;
fftw_plan coeff,coefb;
rfftw_plan rcoeff,rcoefb;

void initfftw()
{
  coeff=fftw_create_plan(N,FFTW_FORWARD,FFTW_MEASURE);
  coefb=fftw_create_plan(N,FFTW_BACKWARD,FFTW_MEASURE);
}

void initglobal()
{

  psi=(comp1D) calloc(N,sizeof(comp));
  psik=(comp1D) calloc(N,sizeof(comp));
  ks=(real1D) calloc(N,sizeof(real));
  kx = (real1D) calloc(N,sizeof(real));
  xm = (real1D) calloc(N,sizeof(real));
  fstruct=(real2D) mymalloc(sizeof(double),10,N/4);
  fstruct2=(real2D) mymalloc(sizeof(double),10,N/4);
}


void init()
{
int ii,jj;
double rad,sqk,theta,mod,r2,xx,yy;
FILE *sortie;
char nom[30];

 srand48(time(0));
 inv=(double) 1./N;
  heure=0.;
  compt=0;
  kx[0]=0.;
  ks[0]=0.;
  for(ii=1;ii<N/2;ii++){
      kx[ii]=2.*pi*ii/N/dx;
      kx[N-ii]=-kx[ii];
	  ks[ii]=0.5*kx[ii]*kx[ii];
	  ks[N-ii]=ks[ii];
  }
  ii=N/2;
  kx[ii]=pi/dx;
  ks[ii]=0.5*kx[ii]*kx[ii];
  

	 for(ii=0;ii<N;ii++){
	 psik[ii].re=0.;
	 psik[ii].im=0.;
	 }
	fftw_one(coefb,psik,psi);
for(ii=0;ii<N/4;ii++)
	for(jj=0;jj<10;jj++){
		fstruct[jj][ii]=0.;
		fstruct2[jj][ii]=0.;
	}
}

void initk(int ind)
{
	int ii;

	for(ii=0;ii<N;ii++){
		psik[ii].re=0.;
		psik[ii].im=0.;
	}
    psik[ind].re=amp;
    psik[0].re=sqrt(1.-amp*amp);
}
