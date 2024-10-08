import seaborn as sns
# sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

# Import pandas library
import pandas as pd

# initialize list of lists

data = [
    [36.58675197969097, 'c1=2, c2=2', 100],
    [38.16827283993194, 'c1=1.4, c2=1.4', 100],
    [35.58675197969097, 'c1=0.5, c2=0.3', 100],
    [36.386198292864776, 'c1=2, c2=2', 50],
    [37.06014399244403, 'c1=1.4, c2=1.4', 50],
    [34.07293726076673, 'c1=0.5, c2=0.3', 50],
    [35.320871337204515, 'c1=2, c2=2', 20],
    [34.49277424839136, 'c1=1.4, c2=1.4', 20],
    [31.678198383279508, 'c1=0.5, c2=0.3', 20],
]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Mean', 'Inertia', 'Particles'])

# print dataframe.
print(df)

g = sns.catplot(
    data=df, kind="bar",
    x="Particles", y="Mean", hue="Inertia",
        edgecolor="0", alpha=0.8, lw=0.4, legend=False,height=5

)
plt.legend(ncol=3, prop={'size': 8.5})

# d = sns.catplot(
#     data=df, kind="point",
#     x="Particles", y="Mean", hue="Inertia", errorbar="sd",
#        height=6, palette="pastel",   markers=["^", "o", "x"], linestyles=["-", "--","dashdot"],
#
# )


# Overlay a stripplot





# g.catplot(data=df, x="Particles", y="Mean", hue="Inertia", kind="point")
# g.despine(left=True)
g.set_axis_labels("No. of Particles", "Sum-Rate(bits/sec/Hz")
plt.ylabel("Sum Rate (bits/sec/Hz)",fontsize=16)

plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
# g.legend.set_title("PSO parameters",prop={'size': 12})
plt.grid(True)
# sns.move_legend(
#     g, loc="upper right", ncol=1, frameon=True, columnspacing=1,
# )
# plt.title('PySwarm Optimization Hyperparameter Performance', fontsize=16)

plt.show()