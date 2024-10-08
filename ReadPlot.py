import pickle
import matplotlib.pyplot as plt


with open('PSOpenalty', 'rb') as f:
    fig = pickle.load(f)

# fig.legend(('y0','y1''y2','y3'))
# fig.legend()

# ax_list = fig.axes
# ax_list[0].get_legend().remove()
# # ax = gca()
# # ax.legend_ = None
# ax_list[0].legend(('IRS between Relay-BS and Device-Relay','No IRS','IRS between Relay-BS only','IRS between Device-Relay only'),  loc='lower right')
plt.xlabel("No. of Iterations",fontsize=16)
plt.ylabel("Sum Rate (bits/sec/Hz)",fontsize=16)
plt.legend( prop={'size': 10}, loc="center right")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()