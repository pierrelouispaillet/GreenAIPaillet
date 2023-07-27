#include "mymalloc.h"
#include <stdio.h>
#include <malloc.h>

void **mymalloc(int size, int n1, int n2)
{
  int i;
  void **p, *p1;
  
  p1 = (void *) calloc(n1 * n2, size);
  if (p1 == NULL)
    {
      printf("Can't allocate mymalloc(%d, %d, %d)\n", size, n1, n2);
    }
  p = (void **) malloc(n1 * sizeof(void *));
  if (p == NULL)
    {
      printf("Can't allocate mymalloc(%d, %d, %d)\n", size, n1, n2);
    }
  p--;
  for (i = 0; i < n1; i++)
    p[i] = (void *) ((long) p1 + n2 * size * i );
  
  return p;
}
