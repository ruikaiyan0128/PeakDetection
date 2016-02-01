#! /Users/ruikaiyan/Python/enter/bin/python3
'''
algorism:

1. every element in the 1D array minus its previous element
   if positive, means it is rising, if negative, means it is falling,
   while one element has possitive value itself and negative value the element
   after it, it means a local maximum

2. the previous 2 days of a local maximum should both rising
   and later 2 days of a local maximum should both falling

3. detect local minimum, the local maximums adjacent to the local minimum should
   larger than the local minimum with a certain threshold

4. if two local maximums have similar amplitude and they are close to each other,
   with the local mininum between them not every small, they should be averaged
   in both amplitude and indices

'''


def detect_peaks(x, mh, threshold, amplitude=False, show=False, ax=None):
    import numpy as np

    # convert to numpy array
    x = np.atleast_1d(x).astype('float64')

    # find indices of all peaks, dx is an array
    dx = x[1:] - x[:-1]

    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    # local maximum indices
    ind_max = []
    ind_min = []
    peak_ind = []
    # detect peaks for the first step
    for i in range(dx.size-1):
        if dx[i] > 0 and dx[i+1] < 0:
            ind_max.append(i+1)

        if dx[i] < 0 and dx[i+1] > 0:
            ind_min.append(i+1)

    ind_max = np.array(ind_max)
    ind_min = np.array(ind_min)

    # first and last values of x cannot be peaks
    if ind_max.size and ind_max[0] == 0:
        ind_max = ind_max[1:]
    if ind_max.size and ind_max[-1] == x.size-1:
        ind_max = ind_max[:-1]

    if ind_min.size and ind_min[0] == 0:
        ind_min = ind_min[1:]
    if ind_min.size and ind_min[-1] == x.size-1:
        ind_min = ind_min[:-1]

    # remove peaks < minimum peak height
    if ind_max.size and mh is not None:
        ind_max = ind_max[x[ind_max] >= mh]
    # print(ind_max)
    # print(ind_min)
    # when the local maximum comes first
    if ind_max[0] < ind_min[0]:
        # determine whether the first local maximum can be a peak
        if x[ind_max[0]] - x[ind_min[0]] > threshold:
            # print('1. %i' % ( ind_max[0]))
            peak_ind.append(ind_max[0])
            # leave the last local maximum to be determined
            for i in range(1, ind_max.size - 1):
                # compare a local maximum to its nearest two local minimums
                if x[ind_max[i]] - x[ind_min[i-1]] > threshold \
                    and x[ind_max[i]] - x[ind_min[i]] > threshold:
                    #print('2. %i' % (ind_max[i]))
                    peak_ind.append(ind_max[i])
                elif x[ind_max[i]] - x[ind_min[i-1]] > threshold \
                        and abs(x[ind_max[i]] - x[ind_max[i+1]]) < 5e8:
                        #print('3. %i' % (ind_max[i]))
                        peak_ind.append(ind_max[i])
                        if x[ind_max[i+1]] - [ind_min[i+1]] < threshold:
                            peak_ind.append(ind_max[i+1])
                elif x[ind_max[i]] - x[ind_min[i]] > threshold \
                        and abs(x[ind_max[i]] - x[ind_max[i-1]]) < 5e8:
                        #print('4. %i' % (ind_max[i]))
                        peak_ind.append(ind_max[i])
                        if x[ind_max[i-1]] - x[ind_min[i-2]] < threshold:
                            peak_ind.append(ind_max[i-1])
            # determine the last peak, if there is no local mininum after it
            if ind_max.size > ind_min.size:
                if x[ind_max[-1]] - x[ind_min[-1]] > threshold:
                    #print('5. %i' % (ind_max[-1]))
                    peak_ind.append(ind_max[-1])
            # if there is another local minimum after it
            elif ind_max.size == ind_min.size:
                if x[ind_max[-1]] - x[ind_min[-2]] > threshold \
                    and x[ind_max[-1]] - x[ind_min[-1]] > threshold:
                    #print('7. %i' % (ind_max[-1]))
                    peak_ind.append(ind_max[-1])
                elif x[ind_max[-1]] - x[ind_min[-1]] > threshold \
                        and abs(x[ind_max[-1]] - x[ind_max[-2]]) < 5e8:
                        #print('8. %i' % (ind_max[-1]))
                        peak_ind.append(ind_max[-1])
                        if x[ind_max[-2]] - x[ind_min[-3]] < threshold:
                            peak_ind.append(ind_max[-2])
    # when local mininum comes first
    elif ind_max[0] > ind_min[0]:
        # leave the last local maximum to be determined
        for i in range(ind_max.size - 1):
            if x[ind_max[i]] - x[ind_min[i]] > threshold \
                and x[ind_max[i]] - x[ind_min[i+1]] > threshold:
                #print('9. %i' % (ind_max[i]))
                peak_ind.append(ind_max[i])
            elif x[ind_max[i]] - x[ind_min[i]] > threshold \
                    and abs(x[ind_max[i]] - x[ind_max[i+1]]) < 5e8:
                    #print('10. %i' % (ind_max[i]))
                    peak_ind.append(ind_max[i])
                    if x[ind_max[i+1]] - x[ind_min[i+2]] < threshold:
                        peak_ind.append(ind_max[i+1])
            elif x[ind_max[i]] - x[ind_min[i+1]] > threshold \
                    and abs(x[ind_max[i]] - x[ind_max[i-1]]) < 5e8:
                    #print('11. %i' % (ind_max[i]))
                    peak_ind.append(ind_max[i])
                    if x[ind_max[i-1]] - x[ind_min[i-1]]:
                        peak_ind.append(ind_max[i-1])
        # if there is no local minimum after it
        if ind_max.size == ind_min.size:
            if x[ind_max[-1]] - x[ind_min[-1]] > threshold:
                #print('12. %i' % (ind_max[-1]))
                peak_ind.append(ind_max[-1])
        elif ind_min.size > ind_max.size:
            if x[ind_max[-1]] - x[ind_min[-2]] > threshold \
                and x[ind_max[-1]] - x[ind_min[-1]] > threshold:
                #print('13. %i' % (ind_max[-1]))
                peak_ind.append(ind_max[-1])
            elif x[ind_max[-1]] - x[ind_min[-1]] > threshold \
                    and abs(x[ind_max[-1]] - x[ind_max[-2]]) < 5e8:
                    #print('14. %i' % (ind_max[-1]))
                    peak_ind.append(ind_max[-1])
                    if x[ind_max[-2]] - x[ind_min[-2]] < threshold:
                        peak_ind.append(ind_max[-2])
    # convert to an array
    peak_ind = sorted(peak_ind)
    peak_ind = np.array(peak_ind)
    amp = [x[i] for i in set(peak_ind)]
    # print(peak_ind)
    if show:
        _plot(x, ax, peak_ind)
    return peak_ind
    if amplitude:
        return amp


def _plot(x, ax, peak_ind):

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print('matplotlib or numpy is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if peak_ind.size:
            label = 'peak'
            label = label + 's' if peak_ind.size > 1 else label
            ax.plot(peak_ind, x[peak_ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (peak_ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Peak detection'
        ax.set_title("%s" % (mode))
        # plt.grid()
        plt.show()
