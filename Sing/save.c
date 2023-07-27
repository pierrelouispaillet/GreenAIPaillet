#include "utilf.h"
#include "global.h"
#include "save.h"
#include "outils.h"



void printx(int ind)
{
int ii,jj,ix,iy;
char nom[256];
float uu,vv;
double s,theta,spec[N];
FILE *fptr;

/*
sprintf(nom,"peak.%d",ind);
fptr=fopen(nom,"wb");
for(ii=0;ii<N;ii++)
	fprintf(fptr,"%g %g %g %g\n",ii*dx,psi[ii].re,psi[ii].im,Car(psi[ii]));
fclose(fptr);
*/
	sprintf(nom,"spectre.%d",ind);
	fptr=fopen(nom,"wb");
	for(ii=1;ii<N/2;ii++)
		fprintf(fptr,"%g %g\n",kx[ii],Car(psik[ii])+Car(psik[N-ii]));
	fclose(fptr);
}

void Structure()
{
int ii,kk,ll;
double rayon,moment[10],mod,mom;
comp1D dpsi,dpsik;
char nom[256];

dpsi=(comp1D) calloc(N,sizeof(comp));
dpsik=(comp1D) calloc(N,sizeof(comp));

for(kk=1;kk<N/4;kk++){
rayon=dx*kk;
for(ll=0;ll<10;ll++)
	moment[ll]=1.;

for(ii=0;ii<N;ii++){
	dpsik[ii].re=2.*psik[ii].re*(cos(kx[ii]*rayon)-1.);
	dpsik[ii].im=2.*psik[ii].im*(cos(kx[ii]*rayon)-1.);
}
fftw_one(coefb,dpsik,dpsi);
for(ii=0;ii<N;ii++){
	mod=sqrt(Car(dpsi[ii]));
	mom=mod;
	for(ll=0;ll<10;ll++){
		fstruct[ll][kk]+=mom*inv;
		mom*=mod;
}
}
}
}

void savestruc()
{
FILE *filestruc;
int ii,ll;

filestruc=fopen("Structure1","w");
for(ii=1;ii<N/4;ii++){
	fprintf(filestruc,"%g ",ii*dx);
	for(ll=0;ll<10;ll++)
		fprintf(filestruc,"%g ",fstruct[ll][ii]/(tmax-tdiss+1.));
	fprintf(filestruc,"\n");
}
fclose(filestruc);
filestruc=fopen("Structure2","w");
for(ii=1;ii<N/4;ii++){
        fprintf(filestruc,"%g ",ii*dx);
        for(ll=0;ll<10;ll++)
                fprintf(filestruc,"%g ",fstruct2[ll][ii]/(tmax-tdiss+1.));
        fprintf(filestruc,"\n");
}
fclose(filestruc);
}
