#! /Users/ruikaiyan/Python/enter/bin/python3
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from PeakDectionPure import *

# use "numdays" days of data
numdays = 50
# the based date, usually today
base = datetime.date(2016, 1, 10)
# a list containing all date string use to produce filename
date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
# the xaxis of the plot, which is from today to the last day of data (61 days later)
xaxis = [base + datetime.timedelta(days=x) for x in range(0, 61)]
# the whole time period from the first available data date to the last date
total_time = sorted(date_list[:-1] + xaxis)

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

# do peak detection for all keys in the dictionary
# store peak indices into dictonary "peak", and peak dates to list "peak_date"
peak = {}
peak_date = []

# three day window and five day window
# create dictionary
td = {}
fd = {}

# prepare the dictionaries with all counts set as 0
for i in range(len(total_time)):
    td[total_time[i]] = 0
    fd[total_time[i]] = 0

# do peak detection for all days
for i in range(numdays):
    # detect indices of peaks for the data from the day (date_list[i]) and store to dictonary peak
    peak[date_list[i]] = detect_peaks(data[date_list[i]], mh=0, threshold=1.2e9, show=False)
    # convert the peak indices to actual dates
    for j in range(len(peak[date_list[i]])):
        peak_date.append(date_list[i] +
                         datetime.timedelta(np.int(peak[date_list[i]][j])))

# count peak times for three days and five days windows
for i in range(len(peak_date)):
    # The first one cannot be valid
    if peak_date[i] > total_time[1]:
        td[peak_date[i]] = td[peak_date[i]] + 1
        td[peak_date[i] - datetime.timedelta(1)] = td[peak_date[i] - datetime.timedelta(1)] + 1
        td[peak_date[i] + datetime.timedelta(1)] = td[peak_date[i] + datetime.timedelta(1)] + 1
    # The first and the second ones cannot be valid
    if peak_date[i] > total_time[2]:
        fd[peak_date[i]] = fd[peak_date[i]] + 1
        fd[peak_date[i] - datetime.timedelta(1)] = fd[peak_date[i] - datetime.timedelta(1)] + 1
        fd[peak_date[i] - datetime.timedelta(2)] = fd[peak_date[i] - datetime.timedelta(2)] + 1
        fd[peak_date[i] + datetime.timedelta(1)] = fd[peak_date[i] + datetime.timedelta(1)] + 1
        fd[peak_date[i] + datetime.timedelta(2)] = fd[peak_date[i] + datetime.timedelta(2)] + 1

# Count times and calculate possibilities of peak dates' appearance
c = sorted([[x, peak_date.count(x)] for x in set(peak_date)])

# Prepare the possibility arrays
yaxis1 = [np.nan] * 61
yaxis2 = [np.nan] * 61
yaxis3 = [np.nan] * 61

# calculate probability
for i in range(len(c)):
    # we only interested in dates after base date, which is usually today
    # when the peak is in the overlapping section, it appears everyday
    if c[i][0] >= base and c[i][0] <= base + datetime.timedelta(61 - numdays):
        c[i][1] = c[i][1] * 100. / numdays
        td[c[i][0]] = td[c[i][0]] * 100. / numdays
        fd[c[i][0]] = fd[c[i][0]] * 100. / numdays

        print('peak at {0}, with a probability of {1:6.2f}, {2:6.2f}, {3:6.2f} %'
              .format(c[i][0], c[i][1], td[c[i][0]], fd[c[i][0]]))

    # for later times, the denominator is the actual days
    elif c[i][0] > base + datetime.timedelta(61-numdays):
        c[i][1] = c[i][1] * 100. / (numdays - ((c[i][0] - base).days - (61 - numdays)))
        td[c[i][0]] = td[c[i][0]] * 100. / (numdays - ((c[i][0] - base).days - (61 - numdays)))
        fd[c[i][0]] = fd[c[i][0]] * 100. / (numdays - ((c[i][0] - base).days - (61 - numdays)))
        print('peak at {0}, with a probability of {1:6.2f}, {2:6.2f}, {3:6.2f} %'
              .format(c[i][0], c[i][1], td[c[i][0]], fd[c[i][0]]))

# insert peak dates' data into xaxis and relate it to yaxis
# loop every date in xaxis first
for i in range(len(xaxis)):
    # then loop peak dates
    for j in range(len(c)):
        # we only care about peaks after base date, which is usually today
        if c[j][0] >= base:
            # when the peak date matches the date on x axis, assign probability to yaxis
            if xaxis[i] == c[j][0]:
                yaxis1[i] = c[j][1]
                yaxis2[i] = td[c[j][0]]
                yaxis3[i] = fd[c[j][0]]
                break
            # when there is no match, set probability to 0
            else:
                yaxis1[i] = 0
                yaxis2[i] = 0
                yaxis3[i] = 0


# Plot data, panel graph and contour plot

font = {'weight': 'bold', 'size': 22}

plt.rc('font', **font)

f, axarr = plt.subplots(3, sharex=True, figsize=(30, 15))
axarr[0].plot(xaxis, yaxis1, linewidth=5, c='red')
axarr[0].set_title('Single day')
axarr[0].set_ylabel('probability %')
axarr[0].grid(True, which='both')
axarr[0].minorticks_on()
axarr[1].plot(xaxis, yaxis2, linewidth=5, c='red')
axarr[1].set_title('3-day window')
axarr[1].set_ylabel('probability %')
axarr[1].grid(True, which='both')
axarr[1].minorticks_on()
axarr[2].plot(xaxis, yaxis3, linewidth=5, c='red')
axarr[2].set_title('5-day window')
axarr[2].set_ylabel('probability %')
axarr[2].grid(True, which='both')
axarr[2].minorticks_on()

#plt.savefig('./0122.eps')
