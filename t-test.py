import pandas as pd
import sys
from scipy import stats

file_path = sys.argv[1]
df_turtle = pd.read_csv(file_path+"/turtle.csv")
df_pet = pd.read_csv(file_path+"/pet.csv")

print("TURTLE")
print('Mean:', 100*df_turtle['acc'].mean())
print('StdDev:', 100*df_turtle['acc'].std())

print("PET")
print('Mean:', 100*df_pet['acc'].mean())
print('StdDev:', 100*df_pet['acc'].std())

print("T-test")
print(stats.ttest_ind(df_turtle['acc'], df_pet['acc'], alternative='less'))