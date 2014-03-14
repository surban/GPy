import numpy as np
import pylab as pb
import GPy
pb.ion()

X1 = 100 * np.random.rand(100)[:,None]
X2 = 100 * np.random.rand(100)[:,None]
#X1.sort()
#X2.sort()

Y1 = np.sin(X1/10.) + np.random.rand(100)[:,None]
Y2 = np.cos(X2/10.) + np.random.rand(100)[:,None]




Mlist = [GPy.kern.Matern32(1,lengthscale=20.,name="Mat")]
kern = GPy.util.multioutput.LCM(input_dim=1,num_outputs=12,kernels_list=Mlist,name='H')


m = GPy.models.GPCoregionalizedRegression(X_list=[X1,X2], Y_list=[Y1,Y2], kernel=kern)
m.optimize()

fig = pb.figure()
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
slices = GPy.util.multioutput.get_slices([Y1,Y2])
m.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],ax=ax0)
m.plot(fixed_inputs=[(1,1)],which_data_rows=slices[1],ax=ax1)
