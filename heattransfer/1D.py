import numpy as np 
import matplotlib.pyplot as plt 
Nz=20; 
T0 = np.zeros(Nz+1); T = np.zeros(Nz+1);

T0[:] = 1.0; Tbc0 = 0.0;  Tbc1 = 0.0 

T0[0] = Tbc0; # left edge
T0[Nz]= Tbc1; # right edge
L=1.0; 
dz = L/Nz; z = np.linspace(0.0, L, Nz); zp= np.linspace(-0.5*dz, L+0.5*dz, Nz+1)
dt=0.001; # set from dt/(dz*dz) < 1/2
dtz2 = dt/dz/dz
def loop_z(T, T0):
  for i in range(T0.size):
    if i==0:
      T[i] = 2.0*Tbc0 - T0[i+1]
    elif i==T0.size-1:
      T[i] = 2.0*Tbc1 - T0[i-1]
    else: 
      T[i] = (1.0 - 2.0*dtz2)*T0[i] + dtz2*(T0[i+1] + T0[i-1]); 
  return
# time loop
for itr in range(1,20):
  loop_z(T, T0)
  T0 = T.copy();
# plot 
plt.plot(zp, T, marker='o');
plt.xlim([0, 1]); plt.ylim([0, 1]); plt.show()
  