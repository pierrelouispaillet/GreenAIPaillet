#include "utilf.h"
#include "global.h"
#include "outils.h"
#include "mymalloc.h"

fftw_complex Cexp(double s)
{
fftw_complex w;

w.re=cos(s);
w.im=sin(s);
return w;
}

fftw_complex Prod(fftw_complex s1,fftw_complex s2)
{
fftw_complex w;

w.re=s1.re*s2.re -s1.im*s2.im;
w.im=s1.re*s2.im+s1.im*s2.re;
return w;
}

double Car(fftw_complex s)
{
double w;

w=s.re*s.re+s.im*s.im;
return w;
}

fftw_complex Pscal(fftw_complex s1,double s2)
{
fftw_complex w;

w.re=s2*s1.re;
w.im=s2*s1.im;
return w;
}

double ImPcc(fftw_complex s1,fftw_complex s2)
{
double w;

w=s1.re*s2.re+s1.im*s2.im;
return w;
}

double Cmasse()
{
  double s;
  int ii,jj;

  s=0.;
  for(ii=0;ii<N;ii++)
      s+=Car(psi[ii]);

  return s*inv;
}

double Cm()
{
	double s;
	int ii;
	
	s=Car(psik[0])+Car(psik[N/2]);
	for(ii=1;ii<N/2;ii++)
		s+=Car(psik[ii])+Car(psik[N-ii]);
	
	return s;
}

double Diss()
{
double w;
int ii,jj;

w=ks[N/2]*ks[N/2]*Car(psik[N/2]);
for(ii=1;ii<N/2;ii++)
  w+=ks[ii]*ks[ii]*(Car(psik[ii])+Car(psik[N-ii]));
return 2.*w*nu;
}

double Ener()
{
double w,s;
int ii,jj;

w=0.;
 for(ii=0;ii<N;ii++){
      s=Car(psi[ii]);
      w+=s*s*s*s/4.;
 }
w=-w*inv;
w+=ks[N/2]*Car(psik[N/2]);
for(ii=1;ii<N/2;ii++)
   w+=ks[ii]*(Car(psik[ii])+Car(psik[N-ii]));
return w;
}

void fonc()
{
	int ii,p,jj,kk,ll;
	double mod,modr,modi,pmod;

for(ii=1;ii<N/4;ii++)
	for(jj=0;jj<N;jj++){
		kk=ii+jj;
		if(kk>=N)
			kk-=N;
		modr=psi[kk].re-psi[jj].re;
		modi=psi[kk].im-psi[jj].im;
		mod=sqrt(modr*modr+modi*modi);
		pmod=mod;
		for(p=0;p<10;p++){
			fstruct[p][ii]+=pmod;
			pmod=pmod*mod;
		}
		ll=jj-ii;
		if(ll<0)
			ll+=N;
		modr+=psi[ll].re-psi[jj].re;
		modi+=psi[ll].im-psi[jj].im;
		               mod=sqrt(modr*modr+modi*modi);
                pmod=mod;
                for(p=0;p<10;p++){
                        fstruct2[p][ii]+=pmod;
                        pmod=pmod*mod;
                }
	}
}
