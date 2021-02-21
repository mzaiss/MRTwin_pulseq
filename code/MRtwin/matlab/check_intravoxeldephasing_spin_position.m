
dim = 12
off = 1 / dim 
sz=10
 [xv, yv ] = meshgrid(linspace(-1+off,1-off,dim), linspace(-1+off,1-off,dim))
 xv=xv/(sz-1);
 yv=yv/(sz-1);
 
 offz=off/(sz-1)
 
figure, mesh(xv,yv,xv+yv); hold on;
mesh(xv-2/(sz-1),yv,xv+yv); hold on;
%  mesh(xv-4/sz,yv,xv+yv)
%  
 
 [xg, yg ] = meshgrid(linspace(-1,1,sz), linspace(-1,1,sz))
 
 mesh(xg,yg,xg+yg); hold on;

 
