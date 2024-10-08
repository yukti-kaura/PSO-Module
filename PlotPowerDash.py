import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt


Full_IRS =    [36.36348702413801, 36.86319493501759,  36.83364726335871, 36.66084863308883, 36.40586629623932,36.30032818248349]
No_IRS =    [22.783486842536,22.763486725409592, 23.068511469438068, 23.00103269924138, 17.819475316844137,15.86479980048059]


IRS_RnB =    [30.64461876905374, 30.799002710362473, 30.724973407974197, 30.64843846674296, 30.62730948829902,30.6436807833988]
IRS_DnR =    [30.528555664344875,30.638616286887995, 30.864620306503035, 30.12867688519042, 30.80629040713682,30.775504707065064]

# Create the pandas DataFrame
Thresholds = [.05, .1, .1,.5,1, 2]

plt.plot(Thresholds, Full_IRS, marker='o', markersize='7',linestyle='dashed', label='Multi-IRS')
plt.plot(Thresholds, IRS_DnR, marker='*',markersize='7', linestyle='dashdot', label='IRS between Device and Relay')
plt.plot(Thresholds, IRS_RnB, marker='^',markersize='7', linestyle='solid', label='IRS between Relay and Base Station')
plt.plot(Thresholds, No_IRS, marker='D',markersize='7', linestyle='dashdot', label='No IRS')
plt.xlim(left=-.005)
plt.grid(True)
plt.ylabel('Mean Sum Rate (Mbps)')
plt.xlabel(r'Relay Power Threshold ($P_{max}$)')
plt.legend()
plt.show()

