
import matplotlib.pyplot as plt

offsets = np.array(coef_xys[0]['y'])
min_off = np.min(offsets)
shifted_offsets = offsets - min_off
linears = np.array(coef_xys[1]['y'])
shifted_linears = linears-1
quads = np.array(coef_xys[2]['y'])
xs = np.array(coef_xys[1]['x'])


fitted_xs = np.arange(2056)
fitted_offsets = quadratic(fitted_xs,*yparametrized_coefs[:,0])
shifted_fitted_offsets = fitted_offsets - np.min(fitted_offsets)
fitted_linears = quadratic(fitted_xs,*yparametrized_coefs[:,1])
shifted_fitted_linears = fitted_linears - 1
fitted_quads = quadratic(fitted_xs,*yparametrized_coefs[:,2])


plt.figure()
plt.title("Offsets")
plt.plot(fitted_xs, fitted_offsets, 'r-',label='fit offset')
plt.plot(xs, offsets,'.',label='offsets')
plt.legend(loc='best')

plt.figure()
plt.title('Linears')
plt.plot(xs,linears,'.',label='linears')
plt.plot(fitted_xs, fitted_linears, 'r-',label='fit linear')
plt.legend(loc='best')

plt.figure()
plt.title("Quads")
plt.plot(fitted_xs, fitted_quads,  'r-', label='fit quad')
plt.plot(xs,quads, '.',label='quads')
plt.legend(loc='best')





plt.figure()
for pixel in [100,1000,2000]:
    plt.plot(fitted_xs, shifted_fitted_offsets + (shifted_fitted_linears*pixel) + (fitted_quads*pixel*pixel),'r-','fit {}'.format(pixel))
    plt.plot(xs, shifted_offsets + (shifted_linears*pixel) + (quads*pixel*pixel), '.',label=str(pixel))
    plt.title("Offset_fits_expanded {}".format(pixel))

plt.legend(loc='best')
plt.show()