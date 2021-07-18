import pandas as pd
import numpy as np

df = pd.read_csv('DeltaGSolv_Matrix_CIGIN_final.csv')

counter_df_columns = {'NAME': df['NAME'], 'Best_solvation_energy': [0 for i in range(28)], 'Second_best_solvation_energy' : [0 for i in range(28)], 'Third_best_solvation_energy' : [0 for i in range(28)]}
counter_df = pd.DataFrame(data=counter_df_columns)

temp_df = df.drop('NAME', axis=1)
for x in temp_df.columns:
    largest = 0
    second_largest = 0
    third_largest = 0
    
    largest_index = 0
    second_largest_index = 0
    third_largest_index = 0
    count = 0
    
    for i in df[x]:
        if i<largest:
            third_largest = second_largest
            third_largest_index = second_largest_index
            second_largest = largest
            second_largest_index = largest_index
            largest = i
            largest_index = count
        elif i<second_largest and i>largest:
            third_largest = second_largest
            third_largest_index = second_largest_index
            second_largest = i
            second_largest_index = count
        elif i<third_largest and i>second_largest and i>largest:
            third_largest = i
            third_largest_index = count
        count += 1
        
    #print(largest_index, second_largest_index, third_largest_index)
    counter_df['Best_solvation_energy'][largest_index] += 1
    counter_df['Second_best_solvation_energy'][second_largest_index] += 1
    counter_df['Third_best_solvation_energy'][third_largest_index] += 1

    
counter_df['Weighted_average'] = [float(0) for i in range(len(counter_df['NAME']))]

for i in range(len(counter_df['NAME'])):
    try:
        counter_df['Weighted_average'][i] = (float(counter_df['Best_solvation_energy'][i]*0.5 + counter_df['Second_best_solvation_energy'][i]*0.3 + counter_df['Third_best_solvation_energy'][i]*0.2))#/float((counter_df['Best_solvation_energy'][i]+counter_df['Second_best_solvation_energy'][i]+counter_df['Third_best_solvation_energy'][i]))
    except:
        continue
    #print((float(counter_df['Best_solvation_energy'][i]*0.5 + counter_df['Second_best_solvation_energy'][i]*0.3 + counter_df['Third_best_solvation_energy'][i]*0.2))/144)

#for i in counter_df['Weighted_average']:
#    counter_df['Weighted_average'][i] = round(counter_df['Weighted_average'][i], 4)

counter_df.to_csv('Weighted_average_df_CIGIN.csv')
