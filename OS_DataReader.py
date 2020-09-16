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

    for i in range(0, files):
        params[i], params_cov[i] = scipy.optimize.curve_fit(Lorentz_func, x_plt[i], y_plt[i], p0 = [0.01,0.1,0.1,x_min], bounds = ([0.01,0,0.1,x_min],[0.05,1,1.5,x_max]), sigma = y_err[i], absolute_sigma = True)
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

    
    #plt.errorbar(x_plt[0], y_plt[0], yerr = None, fmt = 'o', markersize = 1)
    #plt.plot(x_plt[0], Lorentz_func(x_plt[0], params[0][0], params[0][1], params[0][2], params[0][3]))
    #plt.grid()
    #plt.show()
    #plt.clf()

    return params, params_cov, slit_width

def calc_ang_res():

    Grating_150_dir = r"C:\Users\Flo\Desktop\F Praktikum\OS\Daten\150"
    Grating_300_dir = r"C:\Users\Flo\Desktop\F Praktikum\OS\Daten\300"
    Grating_1200_dir = r"C:\Users\Flo\Desktop\F Praktikum\OS\Daten\1200"

    params, params_cov, slit_width = fit_spectrum(Grating_150_dir)

    params1, params_cov1, slit_width1 = fit_spectrum(Grating_300_dir)

    params2, params_cov2, slit_width2 = fit_spectrum(Grating_1200_dir)

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

    plt.errorbar(x_plt, y_plt, yerr = None, fmt = 'x', markersize = 5)
    plt.errorbar(x_plt1, y_plt1, yerr = None, fmt = 'x', markersize = 5)
    plt.errorbar(x_plt2, y_plt2, yerr = None, fmt = 'x', markersize = 5)

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