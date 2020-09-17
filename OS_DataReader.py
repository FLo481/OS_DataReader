import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import re

def Lorentz_func (x, y0, a, Gamma, xc):

    return y0 + a/np.pi*Gamma/(Gamma**2+(x-xc)**2)

def reader (dirName):

    data = {}
    slit_width = [] 
    i = 0
    j = 0
    
    for root, dirs , files in os.walk(dirName):
        for file in files:
            if file.endswith(".asc"):
                slit_width.append(re.findall('\d+', str(file))[1])
                print("Reading in file " + os.path.join(root, file))
                with open(os.path.join(root, file)) as f:
                    reader = csv.reader(f, delimiter = '\t')
                    for row in reader:
                        data[j, i] = row
                        i += 1
                i = 0
                j += 1

    f.close()

    #explanation of indexing

    #data[number of file, number of line][number of entry in line]

    return data, j, slit_width

def plot_spectrum(x_plt, y_plt, y0, a, Gamma, xc, grating, slit_width):

    if slit_width == 0:
        n = 100
    elif slit_width == 1:
        n = 10
    elif slit_width == 2:
        n = 25
    elif slit_width == 3:
        n = 50

    plt.errorbar(x_plt, y_plt, yerr = None, fmt = 'o', markersize = 1)
    plt.title('Grating : {}, Slit width : {}' .format(grating, n))
    plt.plot(x_plt, Lorentz_func(x_plt, y0, a, Gamma, xc), '-b')
    
    plt.grid()
    plt.show()
    plt.clf()

    return 0

def fit_spectrum(dirName):

    length = 1024  
    data, files, slit_width = reader(dirName)
    x_plt = np.empty([files, length], dtype = float)
    y_plt = np.empty([files, length], dtype = float)
    y_err = np.empty([files, length], dtype = float)
    params = [0]*files
    params_cov = [0]*files
    y_max = []
    x_min = 817
    x_max = 821

    for j in range(0, files):
        for i in range(0, length):
            x_plt[j,i] = data[j,i][0]
            y_plt[j,i] = data[j,i][512]
        y_max.append(max(y_plt[j]))
    
    for j in range(0, files):
        for i in range(0, length):
            y_err[j,i] = y_plt[j,i]/np.sqrt(y_plt[j,i])
            y_plt[j,i] = y_plt[j,i]/y_max[j]

    #use different initial values and bounds for different gratings

    if int(re.findall('\d+', str(dirName))[0]) == 150: #good
        intial_values = [0.01,0.1,0.1,x_min]
        bounds = ([0.01,0,0.01,x_min],[0.1,1,2,x_max])
    elif int(re.findall('\d+', str(dirName))[0]) == 300: #good
        intial_values = [0.01,0.1,0.1,x_min]
        bounds = ([0.01,0,0.01,x_min],[0.1,1,2,x_max])
    elif int(re.findall('\d+', str(dirName))[0]) == 1200: #good
        intial_values = [0.01,0.1,0.01,x_min]
        bounds = ([0.01,0,0.01,x_min],[0.1,1,1,x_max])

    for i in range(0, files):
        params[i], params_cov[i] = scipy.optimize.curve_fit(Lorentz_func, x_plt[i], y_plt[i], p0 = intial_values, bounds = bounds, sigma = y_err[i], absolute_sigma = True)
        print("x_c = ", params[i][3], " FWHM = ", 2*params[i][2], " Angular resolution = ", params[i][3]/(2*params[i][2]))
        
    #compute Chi squared if necessary

    #chi_squared = [0]*files
    #fit = np.empty([files, length], dtype = float)

    #for i in range(0, files):
    #    fit[i] = Lorentz_func(x_plt[i], params[i][0], params[i][1], params[i][2], params[i][3])
    
    #for j in range(0, files):
    #    for i in range(0, length):
    #        chi_squared[j] += (y_plt[j,i] - fit[j,i])**2/(y_err[j,i])**2

    #for i in range(0, files):
    #    print("x_c = ", params[i][3], " FWHM = ", 2*params[i][2], " Angular resolution = ", params[i][3]/(2*params[i][2]), "red. chi squared =", chi_squared[i]/(len(x_plt[i])-4))

    return params, params_cov, slit_width, x_plt, y_plt

def calc_ang_res():

    Grating_150_dir = r"C:\Users\Flo\Desktop\F Praktikum\OS\Daten\150"
    Grating_300_dir = r"C:\Users\Flo\Desktop\F Praktikum\OS\Daten\300"
    Grating_1200_dir = r"C:\Users\Flo\Desktop\F Praktikum\OS\Daten\1200"

    params, params_cov, slit_width, x, y = fit_spectrum(Grating_150_dir)

    #n = 1
    #plot_spectrum(x[n], y[n], params[n][0], params[n][1], params[n][2], params[n][3], 150, n)

    params1, params_cov1, slit_width1, x1, y1 = fit_spectrum(Grating_300_dir)

    #n1 = 1
    #plot_spectrum(x1[n1], y1[n1], params1[n1][0], params1[n1][1], params1[n1][2], params1[n1][3], 300, n1)

    params2, params_cov2, slit_width2, x2, y2 = fit_spectrum(Grating_1200_dir)

    #n2 = 2
    #plot_spectrum(x2[n2], y2[n2], params2[n2][0], params2[n2][1], params2[n2][2], params2[n2][3], 1200, n2)



    x_plt = np.empty(len(slit_width), dtype = float)
    y_plt = np.empty(len(slit_width), dtype = float)
    x_plt1 = np.empty(len(slit_width1), dtype = float)
    y_plt1 = np.empty(len(slit_width1), dtype = float)
    x_plt2 = np.empty(len(slit_width2), dtype = float)
    y_plt2 = np.empty(len(slit_width2), dtype = float)

    x_plt[:] = slit_width
    x_plt1[:] = slit_width1
    x_plt2[:] = slit_width2

    for i in range(0, len(slit_width)):
        y_plt[i] = params[i][3]/(2*params[i][2])
        y_plt1[i] = params1[i][3]/(2*params1[i][2])
        y_plt2[i] = params2[i][3]/(2*params2[i][2])

    #plt.errorbar(x_plt, y_plt, yerr = None, fmt = 'x', markersize = 5, label = 'Grating 150')
    #plt.errorbar(x_plt1, y_plt1, yerr = None, fmt = 'x', markersize = 5, label = 'Grating 300')
    #plt.errorbar(x_plt2, y_plt2, yerr = None, fmt = 'x', markersize = 5, label = 'Grating 1200')

    plt.legend()
    plt.grid()
    plt.show()
    plt.clf()

    return 0
   
def main():
    #test = r"C:\Users\Flo\Desktop\F Praktikum\OS\Daten\AV"
    #reader(test)
    #fit_spectrum(test)
    calc_ang_res()

if __name__ == "__main__" :
    main()