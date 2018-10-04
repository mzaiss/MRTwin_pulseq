#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>




void precess( bx,by,bz,mx,my,mz)
double  bx,by,bz,*mx,*my,*mz;
{
  double  b,c,k,s,nx,ny,nz;

  if ( (b = sqrt(bx*bx+by*by+bz*bz)) != 0.0 )
  {
     bx /= b;  nx = *mx;
     by /= b;  ny = *my;
     bz /= b;  nz = *mz;

     c = sin(0.5*b); c = 2.0*c*c;
     s = sin(b);
     k = nx*bx+ny*by+nz*bz;

     *mx += (bx*k-nx)*c + (ny*bz-nz*by)*s;
     *my += (by*k-ny)*c + (nz*bx-nx*bz)*s;
     *mz += (bz*k-nz)*c + (nx*by-ny*bx)*s;
   }
}

void dephase(bz,mx,my,mz)
double  bz,*mx,*my,*mz;
{
  double  b,c,k,s,nx,ny,nz;

  if ( (b = sqrt(bz*bz)) != 0.0 )
  {
               nx = *mx;
               ny = *my;
     bz /= b;  nz = *mz;

     c = sin(0.5*b); c = 2.0*c*c;
     s = sin(b);
     k = nz*bz;

     *mx += (-nx)*c + ny*bz*s;
     *my += (-ny)*c + (-nx*bz)*s;
   }
}

void relax( e1,e2,mx,my,mz)
double  e1,e2,*mx,*my,*mz;
{
  *mx *= e2;
  *my *= e2;
  *mz = 1.0+e1*(*mz-1.0);
}




