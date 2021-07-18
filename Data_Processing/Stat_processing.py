import pandas as pd
import numpy as np

df = pd.read_csv('DeltaGSolv_Matrix_DELFOS_final.csv')

#Avg of Solutes
min_list = ['Min']
max_list = ['Max']
avg_list = ['Avg']

for i in range(1, len(df.columns)):
    sum = float(0)
    min = df[df.columns[i]][0]
    max = df[df.columns[i]][0]
    for j in range(len(df)):
        if df[df.columns[i]][j] < min:
            min = df[df.columns[i]][j]
        if df[df.columns[i]][j] > max:
            max = df[df.columns[i]][j]
        sum += float(df[df.columns[i]][j])
    min_list.append(min)
    max_list.append(max)
    avg_list.append(sum/len(df))
df.loc[28] = min_list
df.loc[29] = max_list
df.loc[30] = avg_list

#Min, Max, Avg of Solvents
df['Min'] = ''
df['Max'] = ''
df['Avg'] = ''
for i in range(len(df)-1):
    sum = float(0)
    max = df.iloc[i][df.columns[1]]
    min = df.iloc[i][df.columns[1]]
    for j in range(1, len(df.columns)-3):
        if float(df.iloc[i][df.columns[j]]) > max:
            max = float(df.iloc[i][df.columns[j]])
        if float(df.iloc[i][df.columns[j]]) < min:
            min = float(df.iloc[i][df.columns[j]])
        sum += float(df.iloc[i][df.columns[j]])
    df['Avg'][i] = sum/(len(df.columns)-3)
    df['Max'][i] = max
    df['Min'][i] = min

df.to_csv('DeltaGSolv_Matrix_DELFOS_stats.csv', index=False)
