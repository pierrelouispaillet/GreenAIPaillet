#include "utilf.h"
#include "global.h"
#include "ini_conditions.h"

void readglob()
{

  if (!intread("N", &N, stdin ))
    {  printf("missing N\n"); exit(1);  }

  if (!intread("tdiss", &tdiss, stdin ))
    {  printf("missing tdiss\n"); exit(1);  }

  if (!intread("tdi", &tdi, stdin ))
    {  printf("missing tdi\n"); exit(1);  }


  if (!intread("tmax", &tmax,stdin ))
    {  printf("missing tmax\n"); exit(1);  }

	if (!intread("tprint", &tprint,stdin ))
    {  printf("missing tprint\n"); exit(1);  }

  if (!doublereadSI("k0", &k0,stdin )) 
    {  printf("missing k0\n"); exit(1);  }

  if (!doublereadSI("nu", &nu,stdin ))
    {  printf("missing nu\n"); exit(1);  }

	if (!doublereadSI("amp", &amp,stdin ))
	{  printf("missing amp\n"); exit(1);  }

	if (!doublereadSI("dx", &dx,stdin ))
      {  printf("missing dx\n"); exit(1);  }

  if (!doublereadSI("dt", &dt,stdin ))
      {  printf("missing dt\n"); exit(1);  }	  

}
