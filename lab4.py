from vpython import *


q = 1.6e-19
particle = sphere(pos=vector(0,3e-7,0), radius=1e-8, color=color.yellow)
particle.v = vector(0, 3e4, 0)
obs01 = sphere(pos=vec(2e-7, 0, 0), radius=particle.radius, color=color.cyan, visible=True, B=vector(0,0,0))
r = obs01.pos-particle.pos
vcross= cross(particle.v,r.hat)
obs01.B = (1E-7*q*vcross)/r.mag**2
print(obs01.B)
attach_arrow(obs01, obs01.B , scale= 1/obs01.B.mag, shaftwidth = obs01.radius, visible = True)
arrow(pos = (obs01.pos),axis = (obs01.B), scale = 5E3, visible = True)