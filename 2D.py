import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # macOS でアニメーション表示用

Nx=20; Ny=20;
T0 = np.zeros([Ny+1, Nx+1]); 
T = np.zeros([Ny+1, Nx+1]);

T0[:] = 0.0
Tbc_w = 1.0;  Tbc_e = 0.0; Tbc_n=0.0; Tbc_s = 0.0; 
T0[:, 0] = Tbc_w; # left edge
T0[:, Nx]= Tbc_e; # right edge
Lx=1.0; Ly=1.0;
dx = Lx/Nx; dy = Ly/Ny;
x = np.linspace(0.0, Lx, Nx); 
xp= np.linspace(-0.5*dx, Lx+0.5*dx, Nx+1) 
y = np.linspace(0.0, Ly, Ny); 
yp= np.linspace(-0.5*dy, Ly+0.5*dy, Ny+1)
dt=0.0005; # set from dt/(dz*dz) < 1/2
dtx2 = dt/dx/dx
dty2 = dt/dy/dy
def update_T(T,T0):
  for j in range(Ny+1): # y-loop
    for i in range(Nx+1): # x-loop
      if i==0: # west
        T[j, i] = 2.0*Tbc_w - T0[j, i+1] # 温度固定
      elif i==Nx: # east
        T[j, i] = 2.0*Tbc_e - T0[j, i-1] 
      elif j==0: # south
        T[j, i] = 2.0*Tbc_s - T0[j+1, i] 
      elif j==Ny: # north  
        T[j, i] = 2.0*Tbc_n - T0[j-1, i] 
      else: 
        T[j, i] = (1.0 - 2.0*dtx2 - 2.0*dty2)*T0[j, i] + dtx2*(T0[j, i+1] + T0[j, i-1]) + dty2*(T0[j+1, i] + T0[j-1, i]); 
  return

# Store all time steps for animation
num_steps = 100
T_history = np.zeros([num_steps, Ny+1, Nx+1])

for itr in range(num_steps): # time looping
  update_T(T,T0)
  T0 = T.copy(); # 更新したTをT0に格納してから，次の時間反復へ 
  T_history[itr] = T0.copy()

# Animation setup
fig, ax = plt.subplots(figsize=(8, 8))
contour = ax.contourf(xp, yp, T_history[0], 20, cmap='viridis', vmin=0.0, vmax=1.0)
ax.set_aspect('equal')
ax.set_xlim([0, Lx])
ax.set_ylim([0, Ly])
ax.set_xlabel('x')
ax.set_ylabel('y')
title = ax.set_title(f'2D Diffusion (t=0.0)')
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Temperature')

def animate(frame):
    ax.clear()
    contour = ax.contourf(xp, yp, T_history[frame], 20, cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_aspect('equal')
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    time = frame * dt
    ax.set_title(f'2D Diffusion (t={time:.4f})')
    return [contour]

anim = FuncAnimation(fig, animate, frames=num_steps, interval=100, blit=False, repeat=True)
plt.show()