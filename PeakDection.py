#! /Users/ruikaiyan/Python/enter/bin/python3
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

# use "numdays" days of data
numdays = 1
base = datetime.date(2015, 12, 10)
date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
xaxis = [base + datetime.timedelta(days=x) for x in range(1, 62)]

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

# read files as arrays
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

# define peak detection function


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    # find indices of all peaks
    # the later one
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)
    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mpd=%d, threshold=%s, edge='%s')"
                     % (mode, mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

# do peak detection for all keys in the dictionary
# store peak indices into dictonary "peak", and peak dates to list "peak_date"
peak = {}
peak_date = []
for i in range(numdays):
    peak[date_list[i]] = detect_peaks(data[date_list[i]], mph=3e9, mpd=5,
                                      edge='both', kpsh=True, show=True)
    # convert the peak indices to actual dates
    for j in range(len(peak[date_list[i]])):
        peak_date.append(date_list[i] +
                         datetime.timedelta(np.int(peak[date_list[i]][j])))


# Count times and calculate possibilities of peak dates' appearance
c = sorted([[x, peak_date.count(x)] for x in set(peak_date)])

yaxis = [np.nan] * 61

for i in range(len(c)):
    if c[i][0] > base and c[i][0] <= base + datetime.timedelta(61 - numdays):
        c[i][1] = c[i][1] * 100. / numdays
        print('peak at {0}, with a probability of {1:6.2f} %'
              .format(c[i][0], c[i][1]))
    elif c[i][0] > base + datetime.timedelta(61-numdays):
        c[i][1] = c[i][1] * 100. / (numdays - ((c[i][0] - base).days - (61 - numdays)))
        print('peak at {0}, with a probability of {1:6.2f} %'
              .format(c[i][0], c[i][1]))


for i in range(len(xaxis)):
    for j in range(len(c)):
        if c[j][0] > base:
            if xaxis[i] == c[j][0]:
                yaxis[i] = c[j][1]
                break
            else:
                yaxis[i] = np.nan

'''
# Do intepolation

df = pd.DataFrame(yaxis, index=xaxis, columns=None)

y = df.interpolate()

y_smoothed = pd.rolling_mean(y, window=5, center=True)

'''
'''
for i in range(len(xaxis)):
    if xaxis[i].day < 10:
        xaxis[i] = datetime.date(month=xaxis[i].month, day=xaxis[i].day, year=0)
    else:
        xaxis[i] = datetime.date(month=xaxis[i].month, day=xaxis[i].day, year=0)

print(xaxis)
print(yaxis)

'''
'''
# Plot data

plt.figure(figsize=(25, 4))
plt.plot(xaxis, y)
plt.ylabel('probability %')
plt.title('Without smoothing')
plt.grid(True, which='both')
plt.minorticks_on()
plt.show()

plt.figure(figsize=(25, 4))
plt.plot(xaxis, y_smoothed)
plt.ylabel('probability %')
plt.minorticks_on()
plt.grid(True, which='both')
plt.title('5-day smoothing')
plt.show()
'''
