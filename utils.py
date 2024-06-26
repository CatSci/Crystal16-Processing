import pandas as pd
import numpy as np
import detecta
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io

def get_columns(dataframe):
  """
  Extracts columns from a DataFrame that partially match strings in a list.

  Args:
      dataframe (pd.DataFrame): The DataFrame to extract columns from.
      columns_to_find (list): A list of strings to match against column names.

  Returns:
      list: A list of column names that partially match the strings in the columns_to_find list.
  """
  columns_to_find = {"Decimal", "Temperature Actual", "Reactor"}
  columns_to_extract = []
  for col in dataframe.columns:
        if any(keyword.lower() in col.lower() and col.lower() != keyword.lower() for keyword in columns_to_find):
            columns_to_extract.append(col)  # Stop searching for the current column after a match is found
  return columns_to_extract



def get_block_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        dataframe (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    index = dataframe[dataframe["Crystal16 Data Report File"] == "Data Block"].index[0]
    raw_data = dataframe.iloc[index :, :].reset_index(drop=True)
    raw_data.columns = raw_data.iloc[0]
    raw_data = raw_data.iloc[1:]
    raw_data.dropna(how='any', inplace=True)
    # Drop rows with any empty strings ('')
    raw_data = raw_data[raw_data.ne('').all(axis=1)]
    columns = get_columns(dataframe= raw_data)
    final_dataframe = raw_data[columns]
    final_dataframe.reset_index(drop= True, inplace= True)

    return final_dataframe

# def get_block_data(dataframe: pd.DataFrame) -> pd.DataFrame:
#     index = dataframe[dataframe["Crystal16 Data Report File"] == "Data Block"].index[0]
#     raw_data = dataframe.iloc[index:].copy()  # Copy to avoid modifying the original DataFrame
#     raw_data.columns = raw_data.iloc[0]
#     raw_data = raw_data.iloc[1:]
#     raw_data = raw_data.dropna(how='any')
#     raw_data = raw_data.loc[:, get_columns(raw_data)]
#     return raw_data

def convert_datatype(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Converts columns of a DataFrame to specified data types.

    Args:
        dataframe (pd.DataFrame): The DataFrame to convert data types.

    Returns:
        pd.DataFrame: The DataFrame with converted data types.
    """
    convert_dict: dict[str, type] = {
        'Decimal Time [mins]': object,  # Assuming you don't want to convert time column
        'Temperature Actual [°C]': float,
        'Reactor1 Transmission [%]': int,
        'Reactor2 Transmission [%]': int,
        'Reactor3 Transmission [%]': int,
        'Reactor4 Transmission [%]': int
    }

    for col, dtype in convert_dict.items():
        try:
            dataframe[col] = dataframe[col].astype(dtype)
        except KeyError:
            pass  # Skip if the column doesn't exist in the DataFrame

    return dataframe

def peak_detection(df, column):
    cloud_clear = {}
    peaks = pd.DataFrame(detecta.detect_onset(df[column], 97, 5))
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
            transmission += [df[column][j]]
            cycle += 1
            
        temp_df = {'Concentration': concs, 'Temps': temps,
                   'Time': times, 'Cycle': sample_cycles, '%': transmission}
        exec(f"{['clear', 'cloud'][i]} = pd.DataFrame(data=temp_df)")
    return locals()['clear'], locals()['cloud'], peaks




def get_clear_and_cloud(dataframe: pd.DataFrame) -> dict:
    """_summary_

    Args:
        dataframe (pd.DataFrame): _description_

    Returns:
        dict: _description_
    """
    clear_dataframe = {}
    cloud_dataframe = {}
    for col in dataframe.columns:
        if "Reactor" in col:
            clear, cloud, peaks = peak_detection(df= dataframe, column= col)
            clear_dataframe[col] = clear
            cloud_dataframe[col] = cloud


    return clear_dataframe, cloud_dataframe


# def get_experiments_details(dataframe: pd.DataFrame) -> pd.DataFrame:
#     """_summary_

#     Args:
#         dataframe (pd.DataFrame): raw data dataframe

#     Returns:
#         pd.DataFrame: extracted dataframe which will have information about reactors
#     """
   
#     exp_index = dataframe[dataframe["Crystal16 Data Report File"] == "Experiment details"].index
#     exp_idx = exp_index.to_list()
#     labjournal_index = dataframe[dataframe["Crystal16 Data Report File"] == "Labjournal"].index
#     labjournal_idx = labjournal_index.to_list()
#     experiment_dataframe = dataframe.iloc[int(exp_idx[0]): labjournal_idx[0], :]
#     # Filter rows where the first column contains "Reactor"
#     reactor_rows = experiment_dataframe[experiment_dataframe.iloc[:, 0].str.contains('Reactor', na=False)]
#     # Extract values from the first two columns
#     extracted_data = reactor_rows.iloc[:, :2]

#     return extracted_data

def get_experiments_details(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extracts details about reactors from the raw data.

    Args:
        dataframe (pd.DataFrame): Raw data DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing information about reactors.
    """
    exp_index = dataframe[dataframe["Crystal16 Data Report File"] == "Experiment details"].index[0]
    labjournal_index = dataframe[dataframe["Crystal16 Data Report File"] == "Labjournal"].index[0]
    experiment_dataframe = dataframe.loc[exp_index + 1: labjournal_index - 1]
    return experiment_dataframe[experiment_dataframe.iloc[:, 0].str.contains('Reactor', na=False)]



# def get_reactor_info(extracted_dataframe: pd.DataFrame) -> dict:
#     """_summary_

#     Args:
#         extracted_dataframe (pd.DataFrame): extracted dataframe which will have information about reactors

#     Returns:
#         dict: reactor information dictionary
#     """
#     reactor_info = {}
#     for i in range(extracted_dataframe.shape[0]):
#         reactor_name = extracted_dataframe.iloc[i, 0]
#         val = extracted_dataframe.iloc[i, 1]
#         conc = val.split('ml')[0] + "ml"
#         solvent = val.split("in")[-1]
#         reactor_info[reactor_name] = {'conc': conc, 'solvent': solvent}

#     return reactor_info


def get_reactor_info(extracted_dataframe: pd.DataFrame) -> dict:
    """Extracts information about reactors from the extracted DataFrame.

    Args:
        extracted_dataframe (pd.DataFrame): Extracted DataFrame containing information about reactors.

    Returns:
        dict: Reactor information dictionary.
    """
    reactor_info = {}
    for index, row in extracted_dataframe.iterrows():
        reactor_name = row.index[0]  # Get the name of the reactor from the index
        val = row.iloc[0]  # Get the value associated with the reactor
        conc = val.split('ml')[0] + "ml"
        solvent = val.split("in")[-1]
        reactor_info[reactor_name] = {'conc': conc, 'solvent': solvent}
    return reactor_info



# def plot_reactor(final_dataframe, clear_dataframe, cloud_dataframe):
#    # Define colors and markers for clear and cloud data
#     marker = 'o'
#     # Create a new figure
#     fig, ax = plt.subplots(figsize=(10, 8))
#     # Plot temperature and transmission for each reactor
#     for reactor_col in final_dataframe.columns:
#         if 'Temperature' in reactor_col:
#             sns.lineplot(data=final_dataframe, x='Decimal Time [mins]', y=reactor_col, color='b')
#         elif 'Transmission' in reactor_col:
#             reactor_num = reactor_col.split()[0][-1]  # Extract reactor number from column name
#             clear_data = clear_dataframe[reactor_col]['Temps']
#             cloud_data = cloud_dataframe[reactor_col]['Temps']
#             clear_time = clear_dataframe[reactor_col]['Time']
#             cloud_time = cloud_dataframe[reactor_col]['Time']
            
#             if reactor_num == '1':
#                 sns.lineplot(data=final_dataframe, x='Decimal Time [mins]', y=reactor_col, color='g', label = reactor_col, ax= ax)
#                 sns.scatterplot(x=clear_time, y=clear_data, color='g', marker=marker, ax= ax)
#                 sns.scatterplot(x=cloud_time, y=cloud_data, color='g', marker=marker, ax= ax)
#             elif reactor_num == '2':
#                 sns.lineplot(data=final_dataframe, x='Decimal Time [mins]', y=reactor_col, color='r', label = reactor_col, ax= ax)
#                 sns.scatterplot(x=clear_time, y=clear_data, color='r', marker=marker, ax= ax)
#                 sns.scatterplot(x=cloud_time, y=cloud_data, color='r', marker=marker, ax= ax)
#             elif reactor_num == '3':
#                 sns.lineplot(data=final_dataframe, x='Decimal Time [mins]', y=reactor_col, color='y', label = reactor_col, ax= ax)
#                 sns.scatterplot(x=clear_time, y=clear_data, color='y', marker=marker, ax= ax)
#                 sns.scatterplot(x=cloud_time, y=cloud_data, color='y', marker=marker, ax= ax)
#             elif reactor_num == '4':
#                 sns.lineplot(data=final_dataframe, x='Decimal Time [mins]', y=reactor_col, color='c', label = reactor_col, ax= ax)
#                 sns.scatterplot(x=clear_time, y=clear_data, color='c', marker=marker, ax= ax)
#                 sns.scatterplot(x=cloud_time, y=cloud_data, color='c', marker=marker, ax= ax)
    
    
 
#     num_data_points = len(final_dataframe['Decimal Time [mins]'])
#     num_ticks = 5
#     ticks_positions = np.linspace(0, num_data_points - 1, num_ticks, dtype=int)
#     ax.set_xticks(final_dataframe['Decimal Time [mins]'][ticks_positions])
#     # Add legend
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), shadow=True, ncol=2)
#     st.pyplot(plt.gcf())  # Pass the current figure (gcf)
#     plot_binary = io.BytesIO()
#     plt.savefig(plot_binary, format='png')
#     plot_binary.seek(0)
#     return plot_binary



def plot_reactor(final_dataframe, clear_dataframe, cloud_dataframe):
    # Define colors for different reactors
    colors = ['g', 'r', 'y', 'c']
    marker = 'o'
    marker_size = 200
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot temperature and transmission for each reactor
    for reactor_col in final_dataframe.columns:
        if 'Temperature' in reactor_col:
            sns.lineplot(data=final_dataframe, x='Decimal Time [mins]', y=reactor_col, color='b', ax=ax)
        elif 'Transmission' in reactor_col:
            reactor_num = int(reactor_col.split()[0][-1]) - 1  # Extract reactor number from column name
            
            # Plot transmission data and scatter points
            sns.lineplot(data=final_dataframe, x='Decimal Time [mins]', y=reactor_col, color=colors[reactor_num], label=reactor_col, ax=ax)
            clear_data = clear_dataframe[reactor_col]['Temps']
            cloud_data = cloud_dataframe[reactor_col]['Temps']
            clear_time = clear_dataframe[reactor_col]['Time']
            cloud_time = cloud_dataframe[reactor_col]['Time']
            sns.scatterplot(x=clear_time, y=clear_data, color=colors[reactor_num], s= marker_size, marker=marker, ax=ax, legend= False)
            sns.scatterplot(x=cloud_time, y=cloud_data, color=colors[reactor_num], s= marker_size, marker=marker, ax=ax, legend = False)
    
    # Set xticks
    num_data_points = len(final_dataframe['Decimal Time [mins]'])
    num_ticks = 5
    ticks_positions = np.linspace(0, num_data_points - 1, num_ticks, dtype=int)
    ax.set_xticks(final_dataframe['Decimal Time [mins]'][ticks_positions])
    
    # Add legend outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), shadow=True, ncol=2)
    
    # Save plot to BytesIO buffer
    plot_binary = io.BytesIO()
    plt.savefig(plot_binary, format='png')
    plot_binary.seek(0)

    st.pyplot(plt.gcf())
    
    return plot_binary