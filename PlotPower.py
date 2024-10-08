import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')
# plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


Full_IRS =    [42.60068981911024, 42.9086749793599,  42.37086983721834, 37.80757463344263, 42.09833435690469,39.514943467508274, 39.514943467508274]
IRS_DnR =    [33.371380500628035,33.48254933725335, 32.585659135967305, 30.053940967453393, 29.253450966954272,32.59176502164965, 32.59176502164965]
IRS_RnB =    [22.543239933003214,  27.853448053394224, 21.491382957072283, 21.048906162871553, 27.726946331929874,26.90895824789373,26.90895824789373]
No_IRS =    [14.09070819126245,19.838892309808354, 15.267846400157037, 15.650806302482133, 21.420484381869166, 20.41311441285213, 20.41311441285213]

# Create the pandas DataFrame
Thresholds = [ .05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]

plt.plot(Thresholds, Full_IRS, marker='o', markersize='10',linestyle='dashed', label='IRS between Device-Relay and Relay-BS')
plt.plot(Thresholds, IRS_RnB, marker='^',markersize='10',  color='mediumvioletred', linestyle='dashed', label='IRS between Relay-BS')
plt.plot(Thresholds, IRS_DnR, marker='*',markersize='10', color='deepskyblue',linestyle='dashdot', label='IRS between Device-Relay')
plt.plot(Thresholds, No_IRS, marker='D',markersize='10', color='orange', linestyle='dashdot', label='No IRS')
# plt.xlim(left=-.005)
plt.grid(True)
plt.ylabel('Sum Rate (Mbps)')
plt.xlabel(r'Device Power Threshold ($P_{max}$)')
plt.legend()
plt.show()

