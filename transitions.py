#! python3
# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import detecta


# %%
# assign clear and cloud points

# Original function - Prints clear and cloud points
# def peak_detection(column):
#     peaks = pd.DataFrame(detecta.detect_onset(column, 95))
#     for i in peaks:
#         cycle = 1
#         for j in peaks[i]:
#             print(f"{['Clear','Cloud'][i]} point, cycle {cycle}: {df['Temp'][j]}")
#             cycle += 1
#     # return pd.DataFrame('Concentration': )


# %%
def peak_detection(column):
    peaks = pd.DataFrame(detecta.detect_onset(column, 97, 5))
    for i in peaks:
        concs = []
        temps = []
        sample_cycles = []
        times = []
        # just for development, not needed later
        transmission = []
        cycle = 1
        for j in peaks[i]:
            # print(f"{['Clear','Cloud'][i]} point, cycle {cycle}: {df['Temp'][j]}")
            # TODO Implement concentration extraction from header
            concs += ['']
            temps += [df['Temperature Actual [°C]'][j]]
            sample_cycles += [cycle]
            # TODO change this to calculate in mins, will need to parse time
            times += [df['Decimal Time [mins]'][j]]
            transmission += [df['Reactor1 Transmission [%]'][j]]
            cycle += 1
        temp_df = {'Concentration': concs, 'Temps': temps,
                   'Time': times, 'Cycle': sample_cycles, '%': transmission}
        exec(f"{['clear', 'cloud'][i]} = pd.DataFrame(data=temp_df)")
    return locals()['clear'], locals()['cloud']


# %%
# import data
df = pd.read_csv(
    r"C:\Users\PhilipLynch\OneDrive - CatSci Ltd\Documents\Python\xstl16\Share to AY and SW\13-14 data block.csv")


# %%
column = df['Reactor1 Transmission [%]']
clear, cloud = peak_detection(column)

# %% Graphing
plt.plot(df['Decimal Time [mins]'], df['Temperature Actual [°C]'], 'b-')
plt.plot(df['Decimal Time [mins]'], df['Reactor1 Transmission [%]'], 'g-')
# add clear and cloud points
plt.plot(clear['Time'], clear['Temps'], 'ro')
plt.plot(cloud['Time'], cloud['Temps'], 'ro')
plt.plot(clear['Time'], clear['%'], 'ko')
plt.plot(cloud['Time'], cloud['%'], 'ko')
plt.show()
