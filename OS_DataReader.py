import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def Lorentz_func (x, y0, a, Gamma, xc):

    return y0 + a/np.pi*Gamma/(Gamma**2+(x-xc)**2)

def reader (dirName):

    data = []
    intensity = []
    wavelength = []
    
    for root, dirs , files in os.walk(dirName):
        for file in files:
            if file.endswith(".asc"):
                print("Reading in file " + os.path.join(root, file))
                with open(os.path.join(root, file)) as f:
                    reader = csv.reader(f, delimiter = '\t')
                    for row in reader:
                        data.append(row)

    f.close()

    #print(data[1023][0])
    #print(data[1][0])
    #print(data[1023][1024])

    #for i in range(0, 1024):
    #    for j in range(1, 1025):
    #        wavelength.append(data[i][0])
    #        intensity.append(data[i][j])

    for i in range(0, 1024):
        wavelength.append(data[i][0])
        intensity.append(data[i][513])




    return intensity, wavelength

def plot_spectrum():

    Angular_res_folder = r"C:\Users\Flo\Desktop\F Praktikum\OS\Daten\AV"
    intensity, wavelength = reader(Angular_res_folder)
    x_min = 817
    x_max = 821
    
    intensity = list(map(float, intensity))
    wavelength = list(map(float, wavelength))

    y_max = max(intensity)

    x_plt = np.empty(len(wavelength), dtype = float)
    y_plt = np.empty(len(intensity), dtype = float)
    x_plt1 = np.empty([], dtype = float)
    y_plt1 = np.empty([], dtype = float)
    y_err = np.empty(len(intensity), dtype = float)
    y_err1 = np.empty([], dtype = float)

    x_plt[:] = wavelength
    y_plt[:] = intensity

    for i in range(0,len(intensity)):
        y_plt[i] = y_plt[i]/y_max
        y_err[i] = y_plt[i]/(np.sqrt(y_plt[i]))

    for i in range(0,len(intensity)):
        if x_max > x_plt[i] > x_min:
            x_plt1 = np.append(x_plt, x_plt[i])
            y_plt1 = np.append(y_plt, y_plt[i])
            y_err1 = np.append(y_err, y_err[i])

    fit = np.empty(len(x_plt1), dtype = float)

    plt.errorbar(x_plt, y_plt, yerr = None, fmt = 'o', markersize = 1)
    params, params_cov = scipy.optimize.curve_fit(Lorentz_func, x_plt1, y_plt1, p0 = [0.04,0.1,0.1,x_min], bounds = ([0.04,0,0,x_min],[0.05,1,1,x_max]), sigma = None, absolute_sigma = True)
    plt.plot(x_plt, Lorentz_func(x_plt, params[0], params[1], params[2], params[3]))

    print("x_c = ", params[3], " FWHM = ", 2*params[2], " Angular resolution = ", params[3]/(2*params[2]))

    #calculation Chi squared

    chi_squared = 0
    fit[:] = Lorentz_func(x_plt1, params[0], params[1], params[2], params[3])

    for i in range(0, len(x_plt1)):
        chi_squared += (y_plt1[i] - fit[i])**2/(y_err1[i])**2

    print("red. Chi squared : ", chi_squared/(len(x_plt1)-4))


    plt.grid()
    #plt.xlim(x_min, x_max)
    #plt.ylim(0, 0.4)

    plt.show()
    plt.clf()

    return 0

def main():
    plot_spectrum()

if __name__ == "__main__" :
    main()