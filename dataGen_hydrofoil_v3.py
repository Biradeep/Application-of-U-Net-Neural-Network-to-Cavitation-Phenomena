################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
# Modified for cavitation
#
# Cavitation number of C_p = (p_a-p_v)*2/(rho U^2)
# U = sqrt(U_x^2 + U_y^2)
# Random variables in this simluation are
#  (a) angle of inclination -10 to 10 degrees
#  (b) shape of hydrofoil
#  (c) p_a fixed
#  
#  python dataGen_hydrofoil_v3.py 1 Cavitationnumber
#
# Makes use of Openfoam_cav
################

import os, math, uuid, sys, random
import numpy as np
import utils 
import matplotlib.pyplot as plt
import getopt

print("The argument values are : ", str(sys.argv))
n = int(sys.argv[1])
print("Index number is ", n)
saved_time='10'
rho=1000;
samples           = 100          #100 no. of datasets to produce
freestream_angle  = 15*math.pi / 180.  # -angle ... angle
freestream_length = 10.           # len * (1. ... factor)
freestream_length_factor = 10.    # length factor
water_vapour_pressure=2400
openfoam_dir="TempFOAM_hydro_"+sys.argv[1]
print("OpenFOAM directory is ",openfoam_dir)
airfoil_database  = "airfoil_database/"
output_dir        = "train_hydro_2021"+"/"	
output_images     = "data_pictures_hydro_2021"+"/"
seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))

def genMesh(airfoilFile):
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0]-1)]))<1e-6:
        ar = ar[:-1]

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(pointIndex, ar[n][0], ar[n][1])
        pointIndex += 1

    print("Current directory is ", os. getcwd()) 
    with open("airfoil_template.geo", "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                outFile.write(line)
    print(os.getcwd())
    os.system('gmsh airfoil.geo -3 -format msh2 -o airfoil.msh')


    if os.system("gmshToFoam airfoil.msh ") != 0:
        print("error during conversion to OpenFoam mesh!")
        return(-1)

    with open("constant/polyMesh/boundary", "rt") as inFile:
        with open("constant/polyMesh/boundaryTemp", "wt") as outFile:
            inBlock = False
            inAerofoil = False
            inTop = False
            inBottom = False
            for line in inFile:
                if "front" in line or "back" in line:
                    inBlock = True
                elif "aerofoil" in line:
                    inAerofoil = True
                elif "top" in line:
                    inTop = True
                elif "bottom" in line:
                    inBottom = True
                if inBlock and "type" in line:
                    line = line.replace("patch", "empty")
                    inBlock = False
                if inAerofoil and "type" in line:
                    line = line.replace("patch", "wall")
                    inAerofoil = False
                if inTop and "type" in line:
                    line = line.replace("patch", "wall")
                    inTop = False
                if inBottom and "type" in line:
                    line = line.replace("patch", "wall")
                    inBottom = False
                outFile.write(line)
    os.rename("constant/polyMesh/boundaryTemp","constant/polyMesh/boundary")

    return(0)

def runSim(freestreamX, freestreamY, freepressure):
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)
    with open("p_rgh_template", "rt") as inFile:
        with open("0/p_rgh", "wt") as outFile:
            for line in inFile:
                line = line.replace("P_AMBIENT", "{}".format(freepressure))
                outFile.write(line)
    os.system("./Allclean")
    os.system("interPhaseChangeFoam")

def outputProcessing(basename, freestreamX, freestreamY, freestream_pressure, dataDir=output_dir, pfile="OpenFOAM_cav/postProcessing/internalCloud/10/somePoints_p_alpha.water.xy", ufile="OpenFOAM_cav/postProcessing/internalCloud/10/somePoints_U.xy", alpha="OpenFOAM_cav/postProcessing/internalCloud/10/somePoints_alpha.xy", res=128, imageIndex=0): 
    # output layout channels:
    # [0] freestream field X + boundary
    # [1] freestream field Y + boundary
    # [2] binary mask for boundary
    # [3] pressure output
    # [4] velocity X output
    # [5] velocity Y output
    # [6] alpha
    npOutput = np.zeros((7, res, res))
    name_directory=openfoam_dir + "/postProcessing/internalCloud/"
    files = os.listdir(name_directory)
    number_files=len(files)
    times=0
    index_max=1
    tmax=0
    if (number_files>1):
        for i in range(0,number_files-1):
            times=float(files[i])
            print(times,i)
            if times>tmax :
                index_max=i
                tmax=times
    else:
        index_max=0
        tmax=float(files[0])

    print("the maximum index is ", index_max)
    print(tmax, files[index_max])
    file_name=name_directory+files[index_max]
#    pfile = file_name + "/somePoints_alpha.water_p.xy" - this is sometimes called something difference for difference
## flavours of openfoam
    alphapfile = file_name + "/somePoints_alpha.water_p.xy"
    ufile = file_name + "/somePoints_U.xy"
    
    ar = np.loadtxt(alphapfile)
    
    print (ar.shape)
    print ("maximum x values is ",  np.amax(ar[:,0]))
    print ("minimum x values is ",  np.amin(ar[:,0]))
    print ("minimum y values is ", np.amin(ar[:,1]))
    print ("minimum y values is ",  np.amax(ar[:,1]))
    pressure=ar[:,4]
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x*1.0 / res - 0.5) * 2 + 0.5
            yf = (y*1.0 / res - 0.5) * 2
            distance=pow(pow(ar[:,0] - xf,2)+pow(ar[:,1] - yf,2),0.5)
            distance_min=distance.min()
            distance_index = np.where(distance == distance_min)
            distance_index=distance_index[0]
            if distance_min<1e-4:
                npOutput[3][x][y] = pressure[distance_index]
                npOutput[0][x][y] = freestreamX
                npOutput[1][x][y] = freestreamY
            else:
                npOutput[3][x][y] = 0
                npOutput[2][x][y] = 1.0

    print("File name is " + ufile)
    ar = np.loadtxt(ufile)
    ux=ar[:,3]
    uy=ar[:,4]
    Xpos=ar[:,0]
    Ypos=ar[:,1]   
    print("The shape of the data is ", ar.shape)
    for y in range(res):
        for x in range(res):    
            curIndex = 0
            Index_agree=-1
            xf = (x*1.0 / res - 0.5) * 2 + 0.5
            yf = (y*1.0 / res - 0.5) * 2
            distance=pow(pow(Xpos - xf,2)+pow(Ypos - yf,2),0.5)

            distance_min=distance.min()
            distance_index = np.where(distance == distance.min())
            distance_index=distance_index[0]

            if distance_min<1e-4:
                npOutput[4][x][y] = ux[distance_index]
                npOutput[5][x][y] = uy[distance_index]
            else:
                npOutput[4][x][y] = 0
                npOutput[5][x][y] = 0


    print("File name is " + alphapfile)
    ar = np.loadtxt(alphapfile)
    alpha=ar[:,3]
    Xpos=ar[:,0]
    Ypos=ar[:,1]   
    print("The shape of the data is ", ar.shape)
    for y in range(res):
        for x in range(res):    
            curIndex = 0
            Index_agree=-1
            xf = (x*1.0 / res - 0.5) * 2 + 0.5
            yf = (y*1.0 / res - 0.5) * 2
            distance=pow(pow(Xpos - xf,2)+pow(Ypos - yf,2),0.5)

            distance_min=distance.min()
            distance_index = np.where(distance == distance.min())
            distance_index=distance_index[0]

            if distance_min<1e-4:
                npOutput[6][x][y] = alpha[distance_index]
            else:
                npOutput[6][x][y] = 0
    fileName = output_dir +"/"  + "%s_%d_%d_%d_%d_%d" % (basename, int(freestreamX*100), int(freestreamY*100), int(freestream_pressure), int(100*Cavitation_number),int(10*180*angle/math.pi) )
    print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, a=npOutput)   ### output the data
    utils.saveAsImage( fileName +   'pressure.png'%(imageIndex), npOutput[3])
    utils.saveAsImage( fileName +   'velX.png'  %(imageIndex), npOutput[4])
    utils.saveAsImage( fileName +   'velY.png'  %(imageIndex), npOutput[5])
    utils.saveAsImage( fileName + 'inputX.png'%(imageIndex), npOutput[0])
    utils.saveAsImage( fileName + 'inputY.png'%(imageIndex), npOutput[1])
    utils.saveAsImage( fileName + 'inputMask.png'%(imageIndex), npOutput[2])	
    utils.saveAsImage( fileName + 'alpha.png'%(imageIndex), npOutput[6])	
  





##
## main
## change the velocity to a velocity
##
## parse n through the call line
## 

U_speed = 2 # m/s  
angle  = np.random.uniform(-freestream_angle, freestream_angle) 
fsX =  math.cos(angle) * U_speed
fsY =  math.sin(angle) * U_speed
Cavitation_number = np.random.uniform(-1,1) 

##
files = os.listdir(airfoil_database)   #airfoil_database)
files.sort()
if len(files)==0:
    print("error - no airfoils found in %s" % airfoil_database)
    exit(1)

utils.makeDirs( ["./data_pictures_hydro_2021", "./" + output_dir, openfoam_dir + "/constant/polyMesh/sets", "./OpenFOAM_cav/constant/polyMesh"] )

print("Run {}:".format(n))

fileNumber = np.random.randint(0, len(files))
basename = os.path.splitext( os.path.basename(files[fileNumber]) )[0]
print("\tusing {}".format(files[fileNumber]))


print("\tUsing angle %+5.3f " %(angle )  )
print("\tResulting freestream vel x,y: {},{}".format(fsX,fsY))
print(os.getcwd())
os.chdir( openfoam_dir )
print(openfoam_dir )    
print(airfoil_database)
testfile="../"+airfoil_database + files[fileNumber]
print(testfile)
genMesh(testfile)
fsPressure=2400+0.5*rho*(U_speed*U_speed)*Cavitation_number

### 2400 + 1000 + np.random.uniform(0 , 3000) 
##  this is a random pressure varying from 3400 to 6400 Pa
runSim(fsX, fsY, fsPressure)

os.chdir("..")

outputProcessing(basename, fsX, fsY, fsPressure, imageIndex=n)
print("\tUsing angle %+5.3f " %(angle )  )
print("\tResulting freestream vel x,y: {},{}".format(fsX,fsY))
print("\tdone")
