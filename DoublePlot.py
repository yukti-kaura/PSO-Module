# import seaborn as sns
# # sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

### Device Power Dash Threshold ###
Full_IRS =    [42.60068981911024, 42.9086749793599,  42.37086983721834, 37.80757463344263, 42.09833435690469,39.514943467508274, 39.514943467508274]
IRS_DnR =    [33.371380500628035,33.48254933725335, 32.585659135967305, 30.053940967453393, 29.253450966954272,32.59176502164965, 32.59176502164965]
IRS_RnB =    [22.543239933003214,  27.853448053394224, 21.491382957072283, 21.048906162871553, 27.726946331929874,26.90895824789373,26.90895824789373]
No_IRS =    [14.09070819126245,19.838892309808354, 15.267846400157037, 15.650806302482133, 21.420484381869166, 20.41311441285213, 20.41311441285213]

# 36.36348702413801,
# 22.783486842536
# 30.64461876905374,
# 30.528555664344875,

Full_IRS1 =    [36.48254003201124 ,39.04418696074336,38.39913899968114,  35.936124031876254, 37.17263575853387,  41.43421132648995,39.514943467508274]
No_IRS1 =    [ 22.7284868132544,22.72848656302711, 22.70348666684639, 22.70348666684639, 22.470936319100286, 21.756552610807713,20.41311441285213]
IRS_RnB1 =    [ 30.432087294854693, 31.476880200128726, 32.05208408911639, 31.725079391988896, 31.512885083020088, 30.710368274910234,26.90895824789373]
IRS_DnR1 =    [32.80040154220005, 31.99834585819367,32.025104590199376, 32.24423409701153, 30.880777097085712, 32.98351409131186,32.59176502164965]
# 36.37921400966245,
# 21.926502691764828,
# 27.28185820049005,
# 30.83165045165851,

# Create the pandas DataFrame
Thresholds = [ .05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]

# Create the figure and the first axes
fig, ax1 = plt.subplots()

# Plot the first dataset on the first axes
# ax1.plot(x1, y1, color='blue')
# ax1.set_xlabel('X1')
# ax1.set_ylabel('Y1')

# Create the secondary x-axis
ax2 = ax1.twiny()

# Plot the second dataset on the secondary x-axis
# ax2.plot(x2, y2, color='red')
ax2.set_xlabel(r'Device Power Threshold ($P_{max}$)')

ax1.plot(Thresholds, Full_IRS1, marker='o', markersize='15',linestyle='dashdot', label=r'IRS between Device-Relay & Relay-BS ($P^\prime_{max}$)',color="mediumvioletred")
ax1.plot(Thresholds, IRS_DnR1, marker='*',markersize='15', linestyle='dashdot', label=r'IRS between Device-Relay ($P^\prime_{max}$)',color="mediumvioletred")
ax1.plot(Thresholds, IRS_RnB1, marker='^',markersize='15', linestyle='dashdot', label=r'IRS between Relay-BS ($P^\prime_{max}$)',color="mediumvioletred")
ax1.plot(Thresholds, No_IRS1, marker='D',markersize='15', linestyle='dashdot', label=r'No IRS ($P^\prime_{max}$)',color="mediumvioletred")
# ax1.set_xlim(left=.1)
# ax2.set_xlim(left=.1)
ax1.set_ylabel('Sum Rate (Mbps)')
ax1.set_xlabel(r'Relay Power Threshold ($P^\prime_{max}$)')
ax2.plot(Thresholds, Full_IRS, marker='o', markersize='15',linestyle='dotted', label=r'IRS between Device-Relay & Relay-BS ($P_{max}$)', color="deepskyblue")
ax2.plot(Thresholds, IRS_DnR, marker='*',markersize='15', linestyle='dotted', label=r'IRS between Device-Relay  ($P_{max}$)', color="deepskyblue")
ax2.plot(Thresholds, IRS_RnB, marker='^',markersize='15', linestyle='dotted', label=r'IRS between Relay-BS ($P_{max}$)', color="deepskyblue")
ax2.plot(Thresholds, No_IRS, marker='D',markersize='10', linestyle='dotted', label=r'No IRS  ($P_{max}$)', color="deepskyblue")
plt.grid(True, linestyle='dotted')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
fig.legend(loc='upper left', bbox_to_anchor=(0.7, 0.9))
# pos = fig.get_position()
# fig.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
# fig.legend( bbox_to_anchor=(1.05, 1.05))
# plt.style.use('bmh')

# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# ax2.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
#
# # Put a legend below current axis
# fig.legend(loc='upper center', bbox_to_anchor=(0.3, -0.05),
#           fancybox=True, shadow=True, ncol=4)
plt.show()

