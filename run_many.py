import os
import time

def loop_method(method_list, nb, loc, n_min, n_max, bg, ell, cen, npt, ell_val, PA_val, bg_val, loga, RA_cen, Dec_cen):
    #start time
    timer = [0]
    timer[0] = time.time()

    cnt=0
    for i in method_list:
        #output_name = loc + str(nb) + "_" + i + "_ellbg.output"
        #output_name = loc + str(nb) + "_" + i + "_ellnobg.output"
        output_name = loc + str(nb) + "_" + i + "_noellnobg.output"
        #output_name = loc + str(nb) + "_" + i + "_noellbg.output"
        if npt > 0:
            output_name = loc + str(nb) + "_" + i + "_ellbgcen_" + str(npt) + ".output"

        if i == 'medsep':
            os.system("python3.5 PROFCL_v1.15.py " + "-man" " --median=o" + " -d Mocks_Mamon" + " --minmax=" + "\'" + str(n_min) + " " + str(n_max) + "\'" + " -o " + output_name + " -D " + "\'M " + str(nb) + " " + str(ell_val) + " " + str(PA_val) + " " + str(bg_val) + " " + str(loga) + " " + str(RA_cen) + " " + str(Dec_cen)  +  "\'")
        else:
            os.system("python3.5 PROFCL_v1.15.py " + "-man" + cen + ell + bg + " -M " + i + " -d Mocks_Mamon" + " -n " + str(npt)  + " --minmax=" + "\'" + str(n_min) + " " + str(n_max) + "\'" + " -o " + output_name + " -D " + "\'M " + str(nb) + " " + str(ell_val) + " " + str(PA_val) + " " + str(bg_val) + " " + str(loga) + " " + str(RA_cen) + " " + str(Dec_cen)  +  "\'")

        timer.append(time.time())
        cnt += 1
    return timer

#meth_list = ['tnc', 'lbb', 'de', 'nm', 'bfgs', 'medsep']
#galnumb_list = [80, 640, 40, 320, 20, 160, 1280]
#galnumb_list = [320, 20, 160, 1280]
galnumb_list = [1280, 640]
meth_list = ['nm']

###cluster min and max###
nmin = 0
nmax = 99

###flag for background###
bg_flag = ""
#bg_flag = " -b"

###flag for ellipticity###
ell_flag = ""
#ell_flag = " -e"

###flag for center###
cen_flag = ""
#cen_flag = " -c"

###number of Monte-Carlo points###
npoints = 0
#npoints = 10000
#npoints = 50000
#npoints = 100000
#npoints = 500000
#npoints = 1000000
#npoints = 200000

###Values of parameters###
e, PA, bg, log_scale_rad, cen_RA, cen_Dec = 0.5, 50, 1, -2.08, 0, 0

###Output files locations###
#loc = "outputs/ell_bg/"
#loc = "outputs/ell_nobg/"
loc = "outputs/noell_nobg/"
#loc = "outputs/noell_bg/"
#loc = "outputs/ell_bg_cen" + str(npoints) + "/"

cnt=0
for i in galnumb_list:
    time1 = loop_method(meth_list, i, loc, nmin, nmax, bg_flag, ell_flag, cen_flag, npoints, e, PA, bg, log_scale_rad, cen_RA, cen_Dec)

    #timing each method separately
    if cnt==0:
        time2 = time1
    else:
        for j in range(len(time1)):
            time2[j] += time1[j]
    cnt += 1

sum = 0
for i in range(len(meth_list)):
    print("Time taken for meth " + meth_list[i] + " is " + str(time2[i+1] - time2[i]) + "s")
    sum += (time2[i+1] - time2[i])
print("Total time taken is ", str(sum) + "s")
