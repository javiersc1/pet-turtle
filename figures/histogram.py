import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def power_law_distribution(size: int, exponent: float):
    """Returns a power law distribution summing up to 1."""
    k = np.arange(1, size + 1)
    power_dist = k ** (-exponent)
    power_dist = power_dist / power_dist.sum()
    return power_dist


powerlaw0 = power_law_distribution(10, 0.0)
powerlaw25 = power_law_distribution(10, 0.25)
powerlaw50 = power_law_distribution(10, 0.5)
powerlaw100 = power_law_distribution(10, 1.0)
powerlaw150 = power_law_distribution(10, 1.5)
powerlaw200 = power_law_distribution(10, 2.0)

# Create a DataFrame for easier plotting with seaborn
df = pd.DataFrame({
    "$\\alpha$ = 0.0": powerlaw0,
    "$\\alpha$ = 0.25": powerlaw25,
    "$\\alpha$ = 0.5": powerlaw50,
    "$\\alpha$ = 1.0": powerlaw100,
    "$\\alpha$ = 1.5": powerlaw150,
    "$\\alpha$ = 2.0": powerlaw200
})

df["Class label"] = np.arange(1, len(df) + 1) # if your x-axis is simply 0,1,2,...

# Convert the DataFrame to "long form" for seaborn
df_long = pd.melt(df, id_vars="Class label", 
                  var_name="Powerlaw", value_name="Frequency")

# Use seaborn's color palette for distinction
sns.set(style="whitegrid")
palette = sns.color_palette("tab10", n_colors=6)

plt.figure(figsize=(6,4))
sns.lineplot(
    data=df_long,
    x="Class label",
    y="Frequency",
    hue="Powerlaw",
    palette=palette,
    style="Powerlaw",
    markers=True,
    markersize=12,
)

plt.xlabel("Class label")
plt.ylabel("Frequency")
plt.xticks(df["Class label"]) 
plt.legend(title="Powerlaw Factor", title_fontsize='13', fontsize='13', loc='upper right')
plt.tight_layout()
plt.savefig("histogram.png", bbox_inches='tight', dpi=600)