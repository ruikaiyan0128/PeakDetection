#! /Users/ruikaiyan/Python/enter/bin/python3

'''
Algorism:
1. create a numpy array for each base day to store 50 days' amplitude
2. find peak and create a new array corresponding to the amplitude array,
   where the peak element is set to 1, others are 0.
3. Then calculate the probability, average amplitude, and perform t test
'''
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from PeakDectionPure import *
import scipy.stats as sp
import math
# use "numdays" days of data each day, and "numdays" days
numdays = 50
# this is the start date, usually today
start_date = datetime.date(2016, 2, 3)
# create the plot axis which contains all dates from "numdays" ago to 61 days later
longxaxis = sorted([start_date - datetime.timedelta(days=x) for x in range(0, numdays)] +
                   [start_date + datetime.timedelta(days=x) for x in range(1, 61)])
# create an array to store probabilitie to plot contour
plotdata1 = np.zeros(shape=(numdays, len(longxaxis)), dtype=np.float)
plotdata3 = np.zeros(shape=(numdays, len(longxaxis)), dtype=np.float)
plotdata5 = np.zeros(shape=(numdays, len(longxaxis)), dtype=np.float)
plotdata_ave = np.zeros(shape=(numdays, len(longxaxis)), dtype=np.float)
plotdata_peak_ave1 = np.zeros(shape=(numdays, len(longxaxis)), dtype=np.float)
plotdata_peak_ave3 = np.zeros(shape=(numdays, len(longxaxis)), dtype=np.float)
plotdata_peak_ave5 = np.zeros(shape=(numdays, len(longxaxis)), dtype=np.float)
pvalue1 = np.ones(shape=(numdays, len(longxaxis)), dtype=np.float)
pvalue3 = np.ones(shape=(numdays, len(longxaxis)), dtype=np.float)
pvalue5 = np.ones(shape=(numdays, len(longxaxis)), dtype=np.float)

#create date label
longxaxislabel = [None] * len(longxaxis)
for i in range(len(longxaxis)):
    longxaxislabel[i] = str(longxaxis[i].month) + '/' + str(longxaxis[i].day)

# loop through numdays
for dt in range(numdays):
    # the based date based on the start_date
    base = start_date - datetime.timedelta(days=dt)
    # a list containing all date string use to produce filename
    date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]

    # the whole time period from the first available data date to the last date
    total_time = sorted(date_list + [base + datetime.timedelta(days=x) for x in range(1, 61)])

    # create arrays to store amplitude
    amplitude = np.zeros(shape=(numdays, len(total_time)), dtype=np.float)
    peak3 = np.zeros(shape=(numdays, len(total_time)), dtype=np.int)
    peak5 = np.zeros(shape=(numdays, len(total_time)), dtype=np.int)
    peak = np.zeros(shape=(numdays, len(total_time)), dtype=np.int)

    # produce the date string and convert to integer
    date_int = []
    for i in range(numdays):
        if date_list[i].month < 10 and date_list[i].day < 10:
            date_int.append(str(date_list[i].year) + '0' + str(date_list[i].month)
            + '0' + str(date_list[i].day))
        elif date_list[i].month < 10 and date_list[i].day >= 10:
            date_int.append(str(date_list[i].year) + '0' + str(date_list[i].month)
            + str(date_list[i].day))
        elif date_list[i].month >= 10 and date_list[i].day < 10:
            date_int.append(str(date_list[i].year) + str(date_list[i].month)
            + '0' + str(date_list[i].day))
        else:
            date_int.append(str(date_list[i].year) + str(date_list[i].month)
            + str(date_list[i].day))

    # sort date so it follows calendar
    date_int = sorted(date_int)
    date_list = sorted(date_list)

    # produce file names to read files
    filename1 = []
    filename2 = []
    filename3 = []
    filename4 = []

    for i in range(numdays):
        filename1.append('ST60N.01.'+date_int[i]+'.dat')
        filename2.append('ST60N.02.'+date_int[i]+'.dat')
        filename3.append('ST60N.03.'+date_int[i]+'.dat')
        filename4.append('ST60N.04.'+date_int[i]+'.dat')

    # read data from files as arrays
    data1 = [None] * numdays
    data2 = [None] * numdays
    data3 = [None] * numdays
    data4 = [None] * numdays

    for i in range(numdays):
        data1[i] = np.fromfile('/Volumes/Data and Files/Indices/'+filename1[i],
                               dtype=np.float32, count=61)
        data2[i] = np.fromfile('/Volumes/Data and Files/Indices/'+filename2[i],
                               dtype=np.float32, count=61)
        data3[i] = np.fromfile('/Volumes/Data and Files/Indices/'+filename3[i],
                               dtype=np.float32, count=61)
        data4[i] = np.fromfile('/Volumes/Data and Files/Indices/'+filename4[i],
                               dtype=np.float32, count=61)

    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    data4 = np.array(data4)

    # average over 4 members
    datave = (data1 + data2 + data3 + data4) / 4.

    # new dictionary with dates being the keys
    data = {}

    # do three-day running mean
    for i in range(numdays):
        data[date_list[i]] = pd.rolling_mean(datave[i], window=5, center=True)

    # do peak detection for all days, notice! date_list is positive sequential
    for i in range(numdays):
        # detect indices of peaks for the data from the day (date_list[i]) and store to dictonary peak
        peak_ind = detect_peaks(data[date_list[i]], mh=0, threshold=1.2e9, show=False)
        for j in peak_ind:
            peak[i, i+j] = 1

        # store amplitude
        data[date_list[i]][0] = datave[i][0]
        data[date_list[i]][1] = datave[i][1]
        data[date_list[i]][-2] = datave[i][-2]
        data[date_list[i]][-1] = datave[i][-1]
        amplitude[i, i: i+61] = data[date_list[i]]

    # complete three and five days' peak arrays
    for i in range(numdays):
        for j in range(1, len(total_time)-1):
            if peak[i, j] == 1:
                peak3[i, j+1] += 1
                peak3[i, j-1] += 1
                peak3[i, j] += 1
        for j in range(2, len(total_time)-2):
            if peak[i, j] == 1:
                peak5[i, j] += 1
                peak5[i, j+1] += 1
                peak5[i, j-1] += 1
                peak5[i, j+1] += 1
                peak5[i, j-1] += 1

    # Prepare the possibility arrays matches the longxaxis
    yaxis1 = np.zeros(len(total_time))
    yaxis3 = np.zeros(len(total_time))
    yaxis5 = np.zeros(len(total_time))

    peak_amplitude_ave1 = np.zeros(len(total_time))
    peak_amplitude_ave3 = np.zeros(len(total_time))
    peak_amplitude_ave5 = np.zeros(len(total_time))

    amplitude_ave = np.zeros(len(total_time))
    # start from base day
    for j in range(numdays - 1, len(total_time)):
        s = 0
        c = 0
        a = []
        b = []
        # store all day average amplitude
        amplitude_ave[j] = sum(amplitude[:, j])/np.count_nonzero(amplitude[:, j])
        # single day's peak probability and average
        for i in range(numdays):
            b.append(amplitude[i, j])
            if peak[i, j] == 1:
                s = s + amplitude[i, j]
                c = c + 1
                a.append(amplitude[i, j])
        # store probability
        if np.count_nonzero(amplitude[:, j]) != 0:
            yaxis1[j] = c / np.count_nonzero(amplitude[:, j])
        # store peak day average amplitude
        if c != 0:
            peak_amplitude_ave1[j] = s / c
        # do t test and store p value to an array
        if np.isnan(sp.ttest_ind(a, b)[1]) == False:
            pvalue1[dt, j-dt] = sp.ttest_ind(a, b)[1]
        # three day
        s = 0
        c = 0
        a = []
        for i in range(numdays):
            if peak3[i, j] >= 1:
                s = s + amplitude[i, j]
                c = c + 1
                a.append(amplitude[i, j])
        # store probability
        if np.count_nonzero(amplitude[:, j]) != 0:
            yaxis3[j] = c / np.count_nonzero(amplitude[:, j])
        # store peak day average amplitude
        if c != 0:
            peak_amplitude_ave3[j] = s / c
        # do t test and store p value to an array
        if np.isnan(sp.ttest_ind(a, b)[1]) == False:
            pvalue3[dt, j-dt] = sp.ttest_ind(a, b)[1]

        # five day
        s = 0
        c = 0
        a = []
        for i in range(numdays):
            if peak5[i, j] >= 1:
                s = s + amplitude[i, j]
                c = c + 1
                a.append(amplitude[i, j])
        # store probability
        if np.count_nonzero(amplitude[:, j]) != 0:
            yaxis5[j] = c / np.count_nonzero(amplitude[:, j])
        # store peak day average amplitude
        if c != 0:
            peak_amplitude_ave5[j] = s / c
        # do t test and store p value to an array
        if np.isnan(sp.ttest_ind(a, b)[1]) == False:
            pvalue5[dt, j-dt] = sp.ttest_ind(a, b)[1]

    # all data starts from base day, store peak probability, and amplitude
    plotdata1[dt, numdays-1-dt: numdays-1-dt+60] = yaxis1[numdays:] * 100
    plotdata3[dt, numdays-1-dt: numdays-1-dt+60] = yaxis3[numdays:] * 100
    plotdata5[dt, numdays-1-dt: numdays-1-dt+60] = yaxis5[numdays:] * 100

    plotdata_ave[dt, numdays - 1 - dt: numdays - 1 + 60 - dt] = amplitude_ave[numdays:]
    plotdata_peak_ave1[dt, numdays - 1 - dt: numdays - 1 + 60 - dt] = peak_amplitude_ave1[numdays:]
    plotdata_peak_ave3[dt, numdays - 1 - dt: numdays - 1 + 60 - dt] = peak_amplitude_ave3[numdays:]
    plotdata_peak_ave5[dt, numdays - 1 - dt: numdays - 1 + 60 - dt] = peak_amplitude_ave5[numdays:]

#print(pvalue3[10, :])
# do contour plot

font = {'weight': 'bold', 'size': 15}
plt.rc('font', **font)

nx, ny = np.shape(plotdata1)
fig = plt.figure(figsize=(30, 15))
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(np.arange(0, 112, 7))
ax.set_yticks(np.arange(0, 51, 7))
ax.set_xticks(np.arange(0, 112, 1), minor=True)
ax.set_yticks(np.arange(0, 51, 1), minor=True)

cmap = plt.cm.get_cmap('jet')
cmap.set_under('white')
cs = plt.pcolor(np.ma.masked_values(plotdata1, 0), vmin=np.spacing(0.0), cmap=cmap)
cb = plt.colorbar(cs)

ax.set_xticklabels(longxaxislabel[::7])
ax.xaxis.set_tick_params(labeltop='on', length=10, width=2, which='major')
ax.xaxis.set_tick_params(length=5, width=1, which='minor')
ax.set_yticklabels(longxaxislabel[numdays-1::-7])
ax.yaxis.set_tick_params(length=10, width=2, which='major')
ax.yaxis.set_tick_params(length=5, width=1, which='minor')
ax.grid(True, linewidth=2)
plt.gca().invert_yaxis()
plt.gca().set_xlim([0, len(longxaxis)])
plt.savefig('./single_day_window_new.eps')

fig = plt.figure(figsize=(30, 15))
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(np.arange(0, 112, 7))
ax.set_yticks(np.arange(0, 51, 7))
ax.set_xticks(np.arange(0, 112, 1), minor=True)
ax.set_yticks(np.arange(0, 51, 1), minor=True)
cs = plt.pcolor(np.ma.masked_values(plotdata3, 0), vmin=np.spacing(0.0), cmap=cmap)
cb = plt.colorbar(cs)
ax.set_xticklabels(longxaxislabel[::7])
ax.xaxis.set_tick_params(labeltop='on', length=10, width=2, which='major')
ax.xaxis.set_tick_params(length=5, width=1, which='minor')
ax.set_yticklabels(longxaxislabel[numdays-1::-7])
ax.yaxis.set_tick_params(length=10, width=2, which='major')
ax.yaxis.set_tick_params(length=5, width=1, which='minor')
ax.grid(True, linewidth=2)
plt.gca().invert_yaxis()
plt.gca().set_xlim([0, len(longxaxis)])
plt.savefig('./three_day_window_new.eps')

fig = plt.figure(figsize=(30, 15))
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(np.arange(0, 112, 7))
ax.set_yticks(np.arange(0, 51, 7))
ax.set_xticks(np.arange(0, 112, 1), minor=True)
ax.set_yticks(np.arange(0, 51, 1), minor=True)
cs = plt.pcolor(np.ma.masked_values(plotdata5, 0), vmin=np.spacing(0.0), cmap=cmap)
cb = plt.colorbar(cs)
ax.set_xticklabels(longxaxislabel[::7])
ax.xaxis.set_tick_params(labeltop='on', length=10, width=2, which='major')
ax.xaxis.set_tick_params(length=5, width=1, which='minor')
ax.set_yticklabels(longxaxislabel[numdays-1::-7])
ax.yaxis.set_tick_params(length=10, width=2, which='major')
ax.yaxis.set_tick_params(length=5, width=1, which='minor')
ax.grid(True, linewidth=2)
plt.gca().invert_yaxis()
plt.gca().set_xlim([0, len(longxaxis)])
plt.savefig('./five_day_window_new.eps')

fig = plt.figure(figsize=(30, 15))
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(np.arange(0, 112, 7))
ax.set_yticks(np.arange(0, 51, 7))
ax.set_xticks(np.arange(0, 112, 1), minor=True)
ax.set_yticks(np.arange(0, 51, 1), minor=True)
cs = plt.pcolor(np.ma.masked_values(plotdata_ave, 0), vmin=np.spacing(0.0), cmap=cmap)
cb = plt.colorbar(cs)
ax.set_xticklabels(longxaxislabel[::7])
ax.xaxis.set_tick_params(labeltop='on', length=10, width=2, which='major')
ax.xaxis.set_tick_params(length=5, width=1, which='minor')
ax.set_yticklabels(longxaxislabel[numdays-1::-7])
ax.yaxis.set_tick_params(length=10, width=2, which='major')
ax.yaxis.set_tick_params(length=5, width=1, which='minor')
ax.grid(True, linewidth=2)
plt.gca().invert_yaxis()
plt.gca().set_xlim([0, len(longxaxis)])
plt.savefig('./overall_amplitude.eps')

fig = plt.figure(figsize=(30, 15))
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(np.arange(0, 112, 7))
ax.set_yticks(np.arange(0, 51, 7))
ax.set_xticks(np.arange(0, 112, 1), minor=True)
ax.set_yticks(np.arange(0, 51, 1), minor=True)
cs = plt.pcolor(np.ma.masked_values(plotdata_peak_ave1, 0), vmin=np.spacing(0.0), cmap=cmap)
cb = plt.colorbar(cs)
ax.set_xticklabels(longxaxislabel[::7])
ax.xaxis.set_tick_params(labeltop='on', length=10, width=2, which='major')
ax.xaxis.set_tick_params(length=5, width=1, which='minor')
ax.set_yticklabels(longxaxislabel[numdays-1::-7])
ax.yaxis.set_tick_params(length=10, width=2, which='major')
ax.yaxis.set_tick_params(length=5, width=1, which='minor')
ax.grid(True, linewidth=2)
plt.gca().invert_yaxis()
plt.gca().set_xlim([0, len(longxaxis)])
plt.gca().set_ylim([0, numdays])
for i in range(numdays):
    for j in range(len(longxaxis)):
        if pvalue1[i, j] < 0.05:
            plt.scatter(x=[j], y=[i], c='black', s=20)
plt.gca().invert_yaxis()
plt.savefig('./single_day_peak_amplitude.eps')

fig = plt.figure(figsize=(30, 15))
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(np.arange(0, 112, 7))
ax.set_yticks(np.arange(0, 51, 7))
ax.set_xticks(np.arange(0, 112, 1), minor=True)
ax.set_yticks(np.arange(0, 51, 1), minor=True)
cs = plt.pcolor(np.ma.masked_values(plotdata_peak_ave3, 0), vmin=np.spacing(0.0), cmap=cmap)
cb = plt.colorbar(cs)
ax.set_xticklabels(longxaxislabel[::7])
ax.xaxis.set_tick_params(labeltop='on', length=10, width=2, which='major')
ax.xaxis.set_tick_params(length=5, width=1, which='minor')
ax.set_yticklabels(longxaxislabel[numdays-1::-7])
ax.yaxis.set_tick_params(length=10, width=2, which='major')
ax.yaxis.set_tick_params(length=5, width=1, which='minor')
ax.grid(True, linewidth=2)
plt.gca().invert_yaxis()
plt.gca().set_xlim([0, len(longxaxis)])
plt.gca().set_ylim([0, numdays])
for i in range(numdays):
    for j in range(len(longxaxis)):
        if pvalue3[i, j] < 0.05:
            plt.scatter(x=[j], y=[i], c='black', s=20)
plt.gca().invert_yaxis()
plt.savefig('./three_day_peak_amplitude.eps')

fig = plt.figure(figsize=(30, 15))
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(np.arange(0, 112, 7))
ax.set_yticks(np.arange(0, 51, 7))
ax.set_xticks(np.arange(0, 112, 1), minor=True)
ax.set_yticks(np.arange(0, 51, 1), minor=True)
cs = plt.pcolor(np.ma.masked_values(plotdata_peak_ave5, 0), vmin=np.spacing(0.0), cmap=cmap)
cb = plt.colorbar(cs)
ax.set_xticklabels(longxaxislabel[::7])
ax.xaxis.set_tick_params(labeltop='on', length=10, width=2, which='major')
ax.xaxis.set_tick_params(length=5, width=1, which='minor')
ax.set_yticklabels(longxaxislabel[numdays-1::-7])
ax.yaxis.set_tick_params(length=10, width=2, which='major')
ax.yaxis.set_tick_params(length=5, width=1, which='minor')
ax.grid(True, linewidth=2)
plt.gca().invert_yaxis()
plt.gca().set_xlim([0, len(longxaxis)])
plt.gca().set_ylim([0, numdays])
for i in range(numdays):
    for j in range(len(longxaxis)):
        if pvalue5[i, j] < 0.05:
            plt.scatter(x=[j], y=[i], c='black', s=20)
plt.gca().invert_yaxis()
plt.savefig('./five_day_peak_amplitude.eps')
