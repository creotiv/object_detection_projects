import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) >= 3:
    d = pd.DataFrame.from_csv(sys.argv[1], index_col=None)
    d['5 seconds'] = (d['time']/(int(sys.argv[2])*100)).astype(int)
    d = d.groupby('5 seconds').sum()
    d = d.drop(['time'], axis=1)
    d.plot()
    plt.show()
else:
    print "Usage: python plot.py [path to the csv report] [number of seconds to group by]"
