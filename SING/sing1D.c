#include "utilf.h"
#include "global.h"
#include "ini_conditions.h"
#include "save.h"
#include "outils.h"
#include "integration.h"
#include <readSIunits.h>


void initialize()
{
  readglob();
  initglobal();
  initfftw();
  init();
}


main(argc,argv)
int argc;
char **argv;
{
  int i,j,k,jd,l,istat,pdf[10000];
  double mt,mta,M0,dmu,dL0,dU0,r12,x12,y12,tempo,mold,max,min,moy;
  double phi2,phi1,ener;
  FILE *sortie;
  double epsmu=1e-6;
  double eps=1e-4;

  initialize();
	sortie=fopen("amp1","wt");
	fclose(sortie);

moy=0.;
for(i=0;i<10000;i++)
	pdf[i]=0;

flux=0.;
dissip=0.;
	for(k=1;k<=tmax;k++){
		for(i=0;i<tprint;i++){
		heure+=dt;
		Intre(0.5*dt);
		Intk(dt);
		Intre(0.5*dt);
//		dissip+=Diss();
/*
		if(k>tmax/2)
			fonc();
*/
if(i%100==0){
	sortie=fopen("amp1","at");
	fprintf(sortie,"%g %g %g %g %g\n",heure,Cmasse(),Ener(),flux/100./dt,dissip/100./dt);
	flux=0.;
	dissip=0.;
	fclose(sortie);
}
if(k>=tdi){
	if(k<tdiss)
	   moy+=Diss()/(tdiss-tdi)/tprint;
	if(k>=tdiss){
		istat=(int) 10.*Diss()/moy;
		if(istat<10000)
		    pdf[istat]++;
}
}
}
printx(k);
if(k>=tdiss)
  fonc();
}
/*
        sortie=fopen("foncs","w");
	for(i=1;i<N/4;i++){
	fprintf(sortie,"%g ",i*dx);
	for(j=0;j<10;j++)
        fprintf(sortie,"%g ",fstruct[j][i]);
	fprintf(sortie,"\n");
	}
        fclose(sortie);
*/
savestruc();
sortie=fopen("pdfdiss","w");
for(i=0;i<10000;i++)
fprintf(sortie,"%g %g\n",moy*(i+0.5)/10.,10.*pdf[i]/(tmax-tdiss+1.)/tprint/moy);
}
