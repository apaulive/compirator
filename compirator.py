#!/usr/bin/env python
# -*- coding: utf-8 -*-

#this program will read in all csv abundance files in a directory and compare other csv abundances in other directories.
#Look into useing Missingno for data visualization for this many species.
import os
import sys #read in arguments
import numpy as np
import pandas
import atexit
from time import time, strftime, localtime
from datetime import timedelta
import seaborn as sns
from matplotlib.colors import LogNorm
#from matplotlib import plt
import matplotlib.pyplot as plt

def heatbringer(dframe,is_id,spec_per_graph):
    nmod = len(is_id) % spec_per_graph
    for i in (range(len(is_id))):
        if (i % spec_per_graph == 0) and (i>=spec_per_graph):
            j = (i-spec_per_graph)
            plt.figure()
            sns.heatmap(dframe[is_id[j:i]], norm=LogNorm())
        elif (i == len(is_id)):
            j = i - nmod
            plt.figure()
            sns.heatmap(dframe[is_id[j:i]], norm=LogNorm())
            #fig.set_yscale('log')
            #plt.clf()
        else:
            continue

def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def log(s, elapsed=None):
    line = "="*40
    print(line)
    print(secondsToStr(), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()

def endlog():
    end = time()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

start = time()
atexit.register(endlog)
log("Start Program")

problem_message = """AIM : Display in log-log the evolution of abundances for a set of species. 
Must be launched in a folder that contains sub-folders, each one containing a simulation.

The script can take various arguments:
(no spaces between the key and the values, only separated by '=')
 * tmax=1.e6 : the end of the output [year]
 * tmin=5e5 : the beginning of the output [year]
 * x=%d : index of the desired spatial point
 * species=CO,H20 : the list of species we want to display /!\ no space !
 * dir=simu0001,simu0002 : the list of sub-folder we want to display /!\ no space !
 * ext=%s : The extension for the output files
 * help : display a little help message on HOW to use various options.

EXAMPLE:
 > nautilus-compare-abundances.py species=CO,H20,C2H6
 > nautilus-compare-abundances.py species=CO,H20,C2H6 dir=simu1,simu2
 > nautilus-compare-abundances.py species=CO,H20,C2H6 x=2
 > nautilus-compare-abundances.py species=CO,H20 tmin=1. tmax=1e6"""

isProblem = False
#standard_directory = 'no_radiolysis'
n = 0

####################################################################
####################    Switches    ################################
####################################################################
#will eventually move manipulable stuff here




for arg in sys.argv[1:]:
  try:
    (key, value) = arg.split("=")
  except:
    key = arg
    value = None
  if (key == 'tmin'):
    t_min = float(value)
  elif (key == 'fac'):
    comp_factor = float(value)
  elif (key == 'tmax'):
    t_max = float(value)
  elif (key == 'dir'):
    directories = value
  elif (key == 'main'):
    standard_directory = value
  elif (key == 'help'):
    isProblem = True
    if (value != None):
      print(value_message % (key, key, value))
  else:
    print("the key '%s' does not match" % key)
    isProblem = True

if isProblem:
  print(problem_message)
  exit()


os.chdir(standard_directory) #assume we start in directory above simulation dir and scripts.
os.chdir('csv')
std_species_list = os.listdir() #all species names, with file extensions
std_species_list.sort() #sort the species names, to keep order consistent
spec_list_std = [] #all species names, without file extensions
nspecies = len(std_species_list) #integer of species in std


#find the timesteps of the simulation, and trim
#timesteps = np.genfromtxt(std_species_list[1], dtype=np.float128, delimiter=',', skip_header=1, usecols=0)
timesteps = pandas.DataFrame(pandas.read_csv(std_species_list[1], sep=',', header=0, names=['Time'], usecols=[0]))
trimtime = pandas.DataFrame()
trimtime = timesteps.drop(timesteps[timesteps['Time'] <= 1000.0].index)
ntrimtime = trimtime.size
std_ab = pandas.DataFrame()
temp = pandas.DataFrame()
comp_ab = pandas.DataFrame()
extra_ab = pandas.DataFrame()
spec_list_comp = []
first_flag = True

for filename in std_species_list:
    species_name = filename.replace('.csv','')
    spec_list_std.append(species_name)
    #temp = pandas.DataFrame(pandas.read_csv(filename, sep=',', header=0, names=['time',filename], usecols=[1]))
    temp = pandas.DataFrame(pandas.read_csv(filename, sep=',', header=0, names=[species_name], usecols=[1], dtype=np.float128))
    std_ab = pandas.concat([std_ab,temp], axis=1)
    temp = pandas.DataFrame(None)
    first_flag = False

#%%
#sort the generated species list
spec_list_std.sort()
#generate species that may be in std, but not comp.
xtra_species_list = []
# changing directories to the comparison directory
os.chdir('../..')
os.chdir(directories)
os.chdir('csv')
#set up the comparsion species list
comp_species_list = os.listdir()
comp_species_list.sort()
comp_species = len(comp_species_list)
first_flag = True

#%%
for filename in comp_species_list:
    species_name = filename.replace('.csv','')
    if species_name in spec_list_std:
        spec_list_comp.append(species_name)
        temp_comp = pandas.DataFrame(pandas.read_csv(filename, sep=',', header=0, names=[species_name], usecols=[1], dtype=np.float128))
        comp_ab = pandas.concat([comp_ab,temp_comp], axis=1)
        temp_comp = pandas.DataFrame(None)
        first_flag = False
    else:
        xtra_species_list.append(species_name)
        
#%%
os.chdir('../..')
spec_list_comp.sort()
xtra_species_list.sort()

#calculate percent difference
diff = pandas.DataFrame(dtype=float)
diff = comp_ab.sub(std_ab)
diff = diff.div(std_ab)

#conditions for comparisons
cond = np.float128(20)
ncond = np.float128(-20)
min_ab = np.float128(1.0e-17)

nd = False
tcol = True
#converting to numpy arrays, too small to work with dataframes
np_diff = diff.to_numpy()
numpy_ab = std_ab.to_numpy()

#iflag and dflag are numpy arrays where True if greater than cond, or less than ncond, and std_ab must be > min_ab
meets_min_np = np.where(numpy_ab >= min_ab, True, False)
iflag_np = np.where(np_diff >= cond, True, False)
dflag_np = np.where(np_diff <= ncond, True, False)
#We then transfer the numpy arrays back to dataframes (which have species labels), and then apply the flags for increase and decrease
meets_min = pandas.DataFrame(meets_min_np, columns=(spec_list_std))
iflag = pandas.DataFrame(iflag_np, columns=(spec_list_std))
dflag = pandas.DataFrame(dflag_np, columns=(spec_list_std))
#print("Iflag before eq")
#print(iflag)
#print("dflag before eq")
#print(dflag)
iflag = iflag.where(cond=(meets_min==True), other=False)
dflag = dflag.where(cond=(meets_min==True), other=False)
#we only take times >1000 yrs
iflag = iflag.drop(range(ntrimtime), axis=0)
dflag = dflag.drop(range(ntrimtime), axis=0)
diff = diff.drop(range(ntrimtime), axis=0)
#we index the remaining times
iflag = iflag.rename(index=trimtime['Time'])
dflag = dflag.rename(index=trimtime['Time'])
diff = diff.rename(index=trimtime['Time'])
#print("iflag after")
#print(iflag)
#print("dflag after")
#print(dflag)

no_rad = True #do you want radiolysis species?
#set up categories
ninc = 0
ndec = 0
is_inc = []
is_dec = []
is_both = []
#increase
is_inc_gas = [] #gas phase neutrals
is_inc_j = [] #surface
is_inc_k = [] #bulk
is_inc_sup = [] #suprathermal
is_inc_cat = [] #cations
is_inc_an = [] #anions
is_inc_gneut =[] #neutrals
#decrease
is_dec_gas = [] #gas phase neutrals
is_dec_j = [] #surface
is_dec_k = [] #bulk
is_dec_sup = [] #suprathermal
is_dec_cat = [] #cations
is_dec_an = [] #anions
is_dec_gneut = [] #neutrals

#%%
for namai in spec_list_std:
    if iflag[namai].any(axis=0) == True:
        if '*' in namai and no_rad == True:
            continue
        elif '*' in namai and no_rad == False:
            is_inc_sup.append(namai)
            is_inc.append(namai)
            ninc += 1
        elif 'J' in namai:
            is_inc_j.append(namai)
            is_inc.append(namai)
            ninc += 1
        elif 'K' in namai:
            is_inc_k.append(namai)
            is_inc.append(namai)
            ninc += 1
        elif namai[-1] == '+':
            is_inc_cat.append(namai)
            is_inc.append(namai)
            ninc += 1
        elif namai[-1] == '-':
            is_inc_an.append(namai)   
            is_inc.append(namai)
            ninc += 1
        elif (namai[-1] != '-') and (namai[-1] != '+') and (namai[0] != 'J') and (namai[0] != 'J'):
            is_inc_gneut.append(namai)
            is_inc.append(namai)
            ninc += 1            
        else:
            is_inc_gas.append(namai)
            is_inc.append(namai)
            ninc += 1
    elif iflag[namai].any(axis=0) == False:
        continue

for namai in spec_list_std:
    if dflag[namai].any(axis=0) == True:
        if '*' in namai and no_rad == True:
            continue
        elif '*' in namai and no_rad == False:
            is_dec_sup.append(namai)
            is_dec.append(namai)
            ndec += 1
        elif 'J' in namai:
            is_dec_j.append(namai)
            is_dec.append(namai)
            ndec += 1
        elif 'K' in namai:
            is_dec_k.append(namai)
            is_dec.append(namai)
            ndec += 1
        elif namai[-1] == '+':
            is_dec_cat.append(namai)
            is_dec.append(namai)
            ndec += 1
        elif namai[-1] == '-':
            is_dec_an.append(namai)   
            is_dec.append(namai)
            ndec += 1
        elif (namai[-1] != '-') and (namai[-1] != '+') and (namai[0] != 'J') and (namai[0] != 'K'):
            is_dec_gneut.append(namai)
            is_dec.append(namai)
            ndec += 1
        else:
            is_dec_gas.append(namai)
            is_dec.append(namai)
            ndec += 1
    elif dflag[namai].any(axis=0) == False:
        continue

for namai in spec_list_std:
    if (namai in is_inc) and (namai in is_dec):
        is_both.append(namai)

#%%        
#graphs? graphs
#set axis to display in scientific notation.
#pandas.set_option('display.float_format','{:.2E}'.format)
nspg = 10 #number of species per graph
#choose what you want to graph
#current is_graph that work are 'increase', 'decrease', or 'all'
is_graph = 'increase'
#condition should be none (for all species).  'cation', 'anion', 'neutral gas', 'gas', 'surface', or 'bulk'.
#make condition for 'simple', 'cc' (carbon chain), 'cyanop' (cyanopolyynes)
condition = 'neutral gas'

#gonna do some heatmaps
#converting to a function

######## PLOT ALL INCREASES AND DECREASES ##############################
if is_graph == 'all':
    if condition == 'none':
        heatbringer(diff, is_inc, nspg)
        heatbringer(diff, is_dec, nspg)
    elif condition == 'neutral gas':
        heatbringer(diff, is_inc_gneut, nspg)
        heatbringer(diff, is_dec_gneut, nspg)
    elif condition == 'cation':
        heatbringer(diff, is_inc_cat, nspg)
        heatbringer(diff, is_dec_cat, nspg)
    elif condition == 'anion':
        heatbringer(diff, is_inc_an, nspg)
        heatbringer(diff, is_dec_an, nspg)
    elif condition == 'gas':
        heatbringer(diff, is_inc_gas, nspg)
        heatbringer(diff, is_dec_gas, nspg)
    elif condition == 'surface':
        heatbringer(diff, is_inc_j, nspg)
        heatbringer(diff, is_dec_j, nspg)
    elif condition == 'bulk':
        heatbringer(diff, is_inc_k, nspg)
        heatbringer(diff, is_dec_k, nspg)
    else:
        print('Check condition please,  should be "none" (for all species).  "cation", "anion", "neutral gas", "gas", "surface", or "bulk"')
        
######   INCREASE ################################

elif is_graph == 'increase':
    if condition == 'none':
        heatbringer(diff, is_inc, nspg)
    elif condition == 'neutral gas':
        heatbringer(diff, is_inc, nspg)
    elif condition == 'cation':
        heatbringer(diff, is_inc_cat, nspg)
    elif condition == 'anion':
        heatbringer(diff, is_inc_an, nspg)
    elif condition == 'gas':
        heatbringer(diff, is_inc_gas, nspg)
    elif condition == 'surface':
        heatbringer(diff, is_inc_j, nspg)
    elif condition == 'bulk':
        heatbringer(diff, is_inc_k, nspg)
    else:
        print('Check condition please,  should be "none" (for all species).  "cation", "anion", "neutral gas", "gas", "surface", or "bulk"')
        
#Decrease ######################################################################################

elif is_graph == 'decrease':
    if condition == 'none':
        heatbringer(diff, is_dec, nspg)
    elif condition == 'neutral gas':
        heatbringer(diff, is_dec_gneut, nspg)
    elif condition == 'cation':
        heatbringer(diff, is_dec_cat, nspg)
    elif condition == 'anion':
        heatbringer(diff, is_dec_an, nspg)
    elif condition == 'gas':
        heatbringer(diff, is_dec_gas, nspg)
    elif condition == 'surface':
        heatbringer(diff, is_dec_j, nspg)
    elif condition == 'bulk':
        heatbringer(diff, is_dec_k, nspg)
    else:
        print('Check condition please,  should be "none" (for all species).  "cation", "anion", "neutral gas", "gas", "surface", or "bulk"')
else:
    print('Check is_graph please, can be "increase", "decrease", or "all"')
                

        #this makes one laaaarge boi
        #plt.subplots(figsize=(20,15))
        #sns.heatmap(diff[is_inc], norm=LogNorm())

#%%
#print(is_inc_cat)
#print(is_inc_j)        
#print(is_inc)
#print(is_dec)
#print(is_both)
print(ninc, "increased out of ", len(spec_list_std))
print(ndec, "decreased out of ", len(spec_list_std))
        


#print(iflag)
#print(dflag)
#print(std_ab)
#print(comp_ab)
#print(diff)
print(endlog())

        


