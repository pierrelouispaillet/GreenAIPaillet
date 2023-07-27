#include "utilf.h"
#include "global.h"
#include "outils.h"
#include "integration.h"



void Intre(double ts)
{
	int ii,jj,kk;
	double mod,modi,x,y,rad;
	fftw_complex cmod;
	
for(ii=0;ii<N;ii++){
		mod=Car(psi[ii]);
		cmod=Cexp(mod*mod*ts);
		psi[ii]=Prod(psi[ii],cmod);
}
}




void Intk(double ts)
{
	int ii,jj;
	double mod,modr,modi;
	fftw_complex cmod;
	
/*
for(ii=0;ii<N;ii++){
 Norm[ii].re=Car(psi[ii]);
 Norm[ii].im=0.;
 }
fftw_one(coeff,Norm,Nk);
*/
fftw_one(coeff,psi,psik);
for(ii=0;ii<N;ii++){
                psik[ii].re *=inv;
                psik[ii].im *=inv;
    }
// flux=0.;

	for(ii=0;ii<N;ii++){
                mod=-ks[ii]*ts;
                cmod=Cexp(mod);
                psik[ii]=Prod(psik[ii],cmod);
		mod=exp(-nu*ks[ii]*ks[ii]*ts);
		dissip+=Car(psik[ii])*(1.-mod*mod);
		psik[ii]=Pscal(psik[ii],mod);
		if(ks[ii]<0.5*k0*k0){
			modr=amp*(drand48()-0.5)*sqrt(ts);
			modi=amp*(drand48()-0.5)*sqrt(ts);
			flux+=2.*(modr*psik[ii].re+modi*psik[ii].im)+modr*modr+modi*modi;
                        psik[ii].re+=modr;
			psik[ii].im+=modi;
	}
               }
fftw_one(coefb,psik,psi);
}

