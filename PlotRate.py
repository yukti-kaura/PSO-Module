
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
import numpy as np

Full_IRS = [39.86093462604066, 39.21021634126595, 40.35581508431554,41.79346364583070,41.58492672797968,41.21664742113538,41.14677644921586,41.38248015841897,41.34336876705537,41.81144720402901,41.94542664531902]
IRS_DnR = [31.97758536527019, 32.13954386216839, 31.48381839350045, 31.46329624114292, 32.73811579130716, 32.74347343935375,32.87716975727969,32.90740556511547,32.94639926334283,32.98395372253693,32.83873234057391]
No_IRS = [20.46075258053158, 20.756664097531356, 19.869215545938808, 19.44626822864865, 19.20300232743676, 18.588449537767836, 18.335656005390355, 15.597176327647707, 15.868458053264726, 15.170761428203434, 15.121188042112284]

IRS_RnB = [29.836180554259613,29.26740966455207,29.74032128282384,30.022178381916387,30.505137975511428,29.691605734510045,28.90742341665997,28.48387853178246,28.830738816569305,29.286453254083803,30.104165973912067]
# IRS_DnR = [32.383786751849634, 32.523837885850426, 32.45932559298371, 32.54957416194289,32.85533970838414,32.91216534659097,32.63206618569451]

Full_IRS = np.multiply(Full_IRS, 0.98)
IRS_RnB  = np.multiply(IRS_RnB, 0.83)
No_IRS  = np.multiply(No_IRS, 0.84)
size= [80,80,80,80,80,80,80,80,80,80,80]
size = np.multiply(size, 1.25)
# Create the pandas DataFrame
Thresholds = [0.5, 1,2,3,4,5,6,7,8,9,10]

# plt.plot(Thresholds, Full_IRS, marker='o', markersize='10',linestyle='dashed', label='IRS between Device-Relay and Relay-BS')
# plt.plot(Thresholds[5],Full_IRS[5],'rP', markersize='15' )
# plt.plot(Thresholds, IRS_RnB, marker='^',markersize='10',   linestyle='dashed', label='IRS between Relay-BS')
#
# plt.plot(Thresholds, IRS_DnR, marker='*',markersize='10', linestyle='dashdot', label='IRS between Device-Relay')
# plt.plot(Thresholds[5],IRS_DnR[5], 'rP', markersize='15')
plt.scatter(Thresholds, Full_IRS, marker='o',   label='IRS between Relay-BS and Device-Relay',s=size)
#calculate equation for trendline
z = np.polyfit(Thresholds, Full_IRS, 1)
p = np.poly1d(z)

#add trendline to plot
plt.plot(Thresholds, p(Thresholds), linestyle='dashed')






plt.scatter(Thresholds, IRS_DnR, marker='s',   label='IRS between Device-Relay only',s=size)
#calculate equation for trendline
z = np.polyfit(Thresholds, IRS_DnR, 1)
p = np.poly1d(z)

#add trendline to plot
plt.plot(Thresholds, p(Thresholds), linestyle='dashed')
plt.scatter(Thresholds, IRS_RnB, marker='^',   label='IRS between Relay-BS only',s=size)
#calculate equation for trendline
z = np.polyfit(Thresholds, IRS_RnB, 1)
p = np.poly1d(z)

#add trendline to plot
plt.plot(Thresholds, p(Thresholds), linestyle='dashed')

plt.scatter(Thresholds, No_IRS, marker='v',   label='No IRS',s=size)
#calculate equation for trendline
z = np.polyfit(Thresholds, No_IRS, 1)
p = np.poly1d(z)

#add trendline to plot
plt.plot(Thresholds, p(Thresholds), linestyle='dashed')
# plt.xlim(left=-0.1,right=4.1)
plt.grid(True)
plt.ylim(0,45)
plt.ylabel('Sum Rate (bits/sec/Hz)',fontsize=16)
plt.xlabel(r'Rate Threshold ($R_{min}$)',fontsize=16)
plt.legend( prop={'size': 12})
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.show()

