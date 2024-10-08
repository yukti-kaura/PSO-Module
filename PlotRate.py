
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

Full_IRS = [38.02775961474328,38.16348191985229,38.863938240499905,38.88464684907108,39.54670737797371,39.9187286566273,39.412343982876095]
No_IRS = [19.36998795769174, 19.657071484945842, 18.855354115548653, 18.529872766860095, 16.47138496510435, 14.038705236470316, 13.277429863597508]
IRS_RnB = [25.949499100245895, 25.587399886834813, 23.961474050977372, 23.29403426644935, 21.05499744137649, 19.812159339590156, 14.79303696407312]
IRS_DnR = [32.383786751849634, 32.523837885850426, 32.45932559298371, 32.54957416194289,32.85533970838414,32.91216534659097,32.63206618569451]



# Create the pandas DataFrame
Thresholds = ['0.5', '1','2','3','4','5','10']

plt.plot(Thresholds, Full_IRS, marker='o', markersize='10',linestyle='dashed', label='IRS between Device-Relay and Relay-BS')
plt.plot(Thresholds[5],Full_IRS[5],'rP', markersize='15' )
plt.plot(Thresholds, IRS_RnB, marker='^',markersize='10',   linestyle='dashed', label='IRS between Relay-BS')

plt.plot(Thresholds, IRS_DnR, marker='*',markersize='10', linestyle='dashdot', label='IRS between Device-Relay')
plt.plot(Thresholds[5],IRS_DnR[5], 'rP', markersize='15')
plt.plot(Thresholds, No_IRS, marker='D',markersize='10',  linestyle='dashdot', label='No IRS')
# plt.xlim(left=-0.1,right=4.1)
plt.grid(True)
plt.ylabel('Sum Rate (bits/sec/Hz)',fontsize=16)
plt.xlabel(r'Rate Threshold ($R_{min}$)',fontsize=16)
plt.legend( prop={'size': 12})
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.show()

