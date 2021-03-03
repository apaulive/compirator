#!/usr/bin/env python
# -*- coding: utf-8 -*-

#this program will read in all csv abundance files in a directory and compare other csv abundances in other directories.
#Look into useing Missingno for data visualization for this many species.
import os
import sys #read in arguments
import numpy as np
import pandas
import atexit
import matplotlib
from time import time, strftime, localtime
from datetime import timedelta
import seaborn as sns
from matplotlib.colors import LogNorm
#from matplotlib import plt
import matplotlib.pyplot as plt

#function to plot heatmaps; calls clean_heat to clean up the seaborn graphs using matplotlib, uses Dataframes
def heatbringer(dframe,is_id,spec_per_graph,ttime):
    nmod = len(is_id) % spec_per_graph
    #display_time = ttime.values.tolist()
    for i in (range(len(is_id))):
        if ((i % spec_per_graph == 0) and (i>=spec_per_graph)) or ((i % spec_per_graph == 0) and (i==len(is_id))):
            j = (i-spec_per_graph)
            """
            fig, ax = plt.subplots()
            im = ax.imshow(dframe[is_id[j:i]])
            #ax.set_yticks(np.arange(len(display_time)
            ax.set_yticks([214,285,357,428,499])
            ax.set_xticks(np.arange(len(is_id[j:i])))
            ax.set_yticklabels(["1.0e3","1.0e4","1.0e5","1.0e6","1.0e7"])
            ax.set_xticklabels(is_id[j:i])
            ax.set_aspect("equal")
            """
            #Feel free to change the cmap color to something that isn'r rainbow road. I like 'cool' and 'plamsa'.
            #'tab20b' is easy to determine groupings, though isn't a continuous gradient
            fig = sns.heatmap(dframe[is_id[j:i]], norm=LogNorm(), yticklabels=100, cmap="gist_rainbow", cbar_kws={'label':'Percent Difference'})
            clean_heat(fig,dframe)
        elif (i == (len(is_id)-1)):
            j = i - nmod
            """
            fig, ax = plt.subplots()
            im = ax.imshow(dframe[is_id[j:i]])
            ax.set_yticks([214,285,357,428,499])
            ax.set_xticks(np.arange(len(is_id[j:i])))
            ax.set_yticklabels(["1.0e3","1.0e4","1.0e5","1.0e6","1.0e7"])
            ax.set_xticklabels(is_id[j:i])
            ax.set_aspect("equal")
            """
            fig = sns.heatmap(dframe[is_id[j:i]], norm=LogNorm(), yticklabels=100, cmap="gist_rainbow", cbar_kws={'label':'Percent Difference'})
            clean_heat(fig,dframe)
        elif (len(is_id) < spec_per_graph):
            print("I am not the clown, I am the entire circus")
            fig = fig = sns.heatmap(dframe[is_id[:]], norm=LogNorm(), yticklabels=100, cmap="gist_rainbow", cbar_kws={'label':'Percent Difference'})
            clean_heat(fig,dframe)
        else:
            continue

def clean_heat(figure,dframe):
    figure.set_xlabel('Species')
    figure.set_ylabel('Time (years)')
    plt.ylim(200,500)
    #plt.yticks(ticks=[200,300,400,500], labels=[200,300,400,500])
    plt.yticks(ticks=[214,285,357,428,499], labels=["1e3","1e4","1e5","1e6","1e7"])    
    plt.show()


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


####################################################################
####################    Switches    ################################
####################################################################
#will eventually move manipulable stuff here
#standard_directory = ''
#directories = ''
#tmin = 1e3



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
trimtime = timesteps.drop(timesteps[timesteps['Time'] <= 999.0].index)
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
    temp = pandas.DataFrame(pandas.read_csv(filename, sep=',', header=0, names=[species_name], usecols=[1], dtype=np.float64))
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
        temp_comp = pandas.DataFrame(pandas.read_csv(filename, sep=',', header=0, names=[species_name], usecols=[1], dtype=np.float64))
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
diff = pandas.DataFrame()
diff = comp_ab.sub(std_ab)
diff = diff.div(std_ab)

#conditions for comparisons
cond = np.float64(20)
ncond = np.float64(-20)
min_ab = np.float64(1.0e-17)

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
#trimdiff = pandas.DataFrame(diff)
#print("Iflag before eq")
#print(iflag)
#print("dflag before eq")
#print(dflag)

#We want to ignore early times, too unstable to flag, generally start checking around 1e3 yrs.
#in order to do so, we make an array the size of the flag, and fill with timevalues
#df_time = pandas.DataFrame(columns=(spec_list_std), index=(range(len(timesteps.index))))
#print(range(len(timesteps)))
#print(df_time)
#print(iflag)
#for s in spec_list_std:
#    df_time[spec_list_std] = timesteps

#print(df_time)

#iflag = iflag.where(cond=(df_time >= 970.0), other=False)
#dflag = dflag.where(cond=(df_time >= 970.0), other=False)
iflag = iflag.where(cond=(meets_min==True), other=False)
dflag = dflag.where(cond=(meets_min==True), other=False)

#we only
#iflag = iflag.drop(range(ntrimtime), axis=0)
#dflag = dflag.drop(range(ntrimtime), axis=0)
#trimdiff = trimdiff.drop(range(ntrimtime), axis=0)

#we index the remaining times
#iflag = iflag.rename(index=trimtime['Time'])
#dflag = dflag.rename(index=trimtime['Time'])
#trimdiff = trimdiff.rename(index=trimtime['Time'])

#We assign the rows index to a time
iflag = iflag.rename(index=timesteps['Time'])
dflag = dflag.rename(index=timesteps['Time'])
diff = diff.rename(index=timesteps['Time'])

#we only flag times after 1000 years
#iflag = iflag.where(cond=(iflag.index >= 1000), other=False)

#print("iflag after")
#print(iflag.index)
#print("dflag after")
#print(dflag.index)



#start of the graphing section
no_rad = True #do you want suprathermal species?
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
#for testing

nspg = 10 #number of species per graph
#choose what you want to graph
#current is_graph that work are 'increase', 'decrease', 'common' or 'all'
is_graph = 'all'
#condition should be none (for all species).  'cation', 'anion', 'neutral gas', 'gas', 'surface', or 'bulk'.
#if condition is 'simple', 'cc' (carbon chain), 'cyanop' (cyanopolyynes),'inorganic', 'coms', or 'ice', will graph the preset examples.
#if condition is 'all' will graph all presets
condition = 'none'
simple = ['H2','CO','H2O','H3+','HCN','HNC','NH3']
cc = ['C4S','HNCCC','C6H','C6H-','HC10N+']
coms = ['CH3OH','CH3OCH3','HCOOCH3','CH2CHCN']
cyanop = []
ice = ['JH2O','JCO2','JCO','JCH3OH','JCH4','JNH3','JOCN','KH2O','KCO2','KCO','KCH3OH','KCH4','KNH3','KOCN']

#gonna do some heatmaps
#because logs throw fits, take log of the time.
log_time = np.log10(timesteps)


######## PLOT ALL INCREASES AND DECREASES ##############################
if is_graph == 'all':
    if condition == 'none':   
        heatbringer(diff, is_inc, nspg, log_time)
        heatbringer(diff, is_dec, nspg, log_time)
    elif condition == 'neutral gas':
        heatbringer(diff, is_inc_gneut, nspg, log_time)
        heatbringer(diff, is_dec_gneut, nspg, log_time)
    elif condition == 'cation':
        heatbringer(diff, is_inc_cat, nspg, log_time)
        heatbringer(diff, is_dec_cat, nspg, log_time)
    elif condition == 'anion':
        heatbringer(diff, is_inc_an, nspg, log_time)
        heatbringer(diff, is_dec_an, nspg, log_time)
    elif condition == 'gas':
        heatbringer(diff, is_inc_gas, nspg, log_time)
        heatbringer(diff, is_dec_gas, nspg, log_time)
    elif condition == 'surface':
        heatbringer(diff, is_inc_j, nspg, log_time)
        heatbringer(diff, is_dec_j, nspg, log_time)
    elif condition == 'bulk':
        heatbringer(diff, is_inc_k, nspg, log_time)
        heatbringer(diff, is_dec_k, nspg, log_time)
    else:
        print('Check condition please,  should be "none" (for all species).  "cation", "anion", "neutral gas", "gas", "surface", or "bulk"')
        
######   INCREASE ################################

elif is_graph == 'increase':
    if condition == 'none':
        heatbringer(diff, is_inc, nspg, log_time)
    elif condition == 'neutral gas':
        heatbringer(diff, is_inc_gneut, nspg, log_time)
    elif condition == 'cation':
        heatbringer(diff, is_inc_cat, nspg, log_time)
    elif condition == 'anion':
        heatbringer(diff, is_inc_an, nspg, log_time)
    elif condition == 'gas':
        heatbringer(diff, is_inc_gas, nspg, log_time)
    elif condition == 'surface':
        heatbringer(diff, is_inc_j, nspg, log_time)
    elif condition == 'bulk':
        heatbringer(diff, is_inc_k, nspg, log_time)
    else:
        print('Check condition please,  should be "none" (for all species).  "cation", "anion", "neutral gas", "gas", "surface", or "bulk"')
        
#Decrease ######################################################################################

elif is_graph == 'decrease':
    if condition == 'none':
        heatbringer(diff, is_dec, nspg, log_time)
    elif condition == 'neutral gas':
        heatbringer(diff, is_dec_gneut, nspg, log_time)
    elif condition == 'cation':
        heatbringer(diff, is_dec_cat, nspg, log_time)
    elif condition == 'anion':
        heatbringer(diff, is_dec_an, nspg, log_time)
    elif condition == 'gas':
        heatbringer(diff, is_dec_gas, nspg, log_time)
    elif condition == 'surface':
        heatbringer(diff, is_dec_j, nspg, log_time)
    elif condition == 'bulk':
        heatbringer(diff, is_dec_k, nspg, log_time)
    else:
        print('Check condition please,  should be "none" (for all species).  "cation", "anion", "neutral gas", "gas", "surface", or "bulk"')
####Common Molecule sets
elif is_graph == 'common':
    if condition == 'all':
        simple = simple + cc + coms + ice
        heatbringer(diff,simple,nspg,log_time)
    elif condition == 'simple':
        heatbringer(diff,simple,nspg,log_time)
    elif condition == 'cc':
        heatbringer(diff,cc,nspg,log_time)
    elif condition == 'coms':
        heatbringer(diff,coms,nspg,log_time)
    elif condition == 'ice':
        heatbringer(diff,ice,nspg,log_time)
    else:
        print('Check condition please,  should be "none" (for all species).  "cation", "anion", "neutral gas", "gas", "surface", or "bulk"')

else:
    print('Check is_graph please, can be "increase", "decrease", "common", or "all"')
                

        #this makes one laaaarge boi
        #plt.subplots(figsize=(20,15))
        #sns.heatmap(diff[is_inc], norm=LogNorm())
      


#%%
#print(is_inc_cat)
#print(is_inc_j)        
#print(is_inc)
#print(is_dec)
#print(is_both)
#print(ninc, "increased out of ", len(spec_list_std))
#print(ndec, "decreased out of ", len(spec_list_std))
        


#print(iflag)
#print(dflag)
#print(std_ab)
#print(comp_ab)
#print(diff)
print(endlog())

        


