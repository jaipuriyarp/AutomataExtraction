import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import numpy as np
import argparse

# lang = 3
def read_csv(file:str) -> pd.DataFrame:
    if not os.path.exists(file):
        raise Exception("file doesn't exist")
    return pd.read_csv(file)


def plot_overview_graph_from_df_list(df_list: list, roundOffxaxis=True):
    if roundOffxaxis==False:
        fig, ax = plt.subplots(figsize=(15, 6))
    else:
        fig, ax = plt.subplots()

    column_name = df_list[0].columns

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']
    # markers = ['o', 'x']

    colors_idx = 0
    for i in range(0,len(df_list),2):
        print(f"Plotting: graph number {i}")
        if not df_list[i].columns.equals(column_name):
            raise Exception(f"The columns name of the two data frame are different. df1: {df_list[i].columns} and cols saved: {columns_name}")


        ax.scatter(df_list[i][column_name[0]], df_list[i][column_name[1]], label='Language' + str(i+1), color=colors[colors_idx],
                marker='o')

        ax.scatter(df_list[i+1][column_name[0]], df_list[i+1][column_name[1]], label='Language' + str(i+1), color=colors[colors_idx],
            marker='x')

        colors_idx += 1

    # Set labels and title
    ax.set_xlabel('Time (s)')
    if 'before' in (column_name[1]):
        ax.set_ylabel("Number of sequences checked", wrap=True)
    else:
        ax.set_ylabel(column_name[1])
    ax.set_title('Nominal active learning approach vs. Random testing')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if roundOffxaxis==False:
        # ax.tick_params(axis='x', which='both', length=10, width=2)
        # ax.set_xlim(0, 3600)
        # Generate fine-grained x-axis tick positions and labels
        fine_grained_ticks = np.arange(0, df1[column_name[0]].max() + 1, 100)  # Adjust the step size as needed
        fine_grained_labels = [str(round(t, 1)) for t in fine_grained_ticks]

        # Set the x-axis tick positions and labels
        ax.set_xticks(fine_grained_ticks)
        ax.set_xticklabels(fine_grained_labels, rotation=45)  # Rotate labels for better readability

    # Display a legend to differentiate the datasets
    # ax.legend(loc='lower right')
    ax.legend(loc='lower right')


    # Show the plot
    plt.show()

def plot_dataframes_in_subplots(df_list: list):
    num_plots = int(len(df_list)/2)
    column_name = df_list[0].columns
    # Determine the number of rows and columns for the subplots (2x4 for 8 subplots)
    # num_rows = 2
    # num_cols = int(num_plots/2) # 4

    num_rows = 3
    num_cols = 3

    # Create a figure with subplots
    # fig, ax = plt.subplots(num_rows, num_cols, figsize=(8.27, 11.69), constrained_layout=True)  # A4 paper size
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(6, 6), constrained_layout=True)

    df_idx = 0
    for i in range(num_plots):
        # Calculate the row and column for the current subplot
        row = i // num_cols
        col = i % num_cols

        # Get the current DataFrame
        df1 = df_list[df_idx]
        df2 = df_list[df_idx+1]

        ax[row, col].scatter(df2[column_name[0]], df2[column_name[1]], label='Random testing',
                             color='red', marker='x')

        ax[row, col].scatter(df1[column_name[0]], df1[column_name[1]], label='Nominal active learning',
                             color='blue', marker='o')

        df_idx += 2

        ax[row, col].set_xlabel('Time (s)')
        if 'before' in (column_name[1]):
            ax[row, col].set_ylabel("Number of sequences checked", wrap=True)
        else:
            ax[row, col].set_ylabel(column_name[1])

        # ax[row, col].legend(loc='lower right')

        ax[row, col].set_title(f'Language {i + 1}')

        # fig.suptitle('Language' + str(i + 1), y=1.02)

        # # Plot the data from the DataFrame on the current subplot
        # axes[row, col].plot(df['column_A'], df['column_B'])
        #
        # # Set title for the subplot (you can customize this)
        # axes[row, col].set_title(f'Dataframe {i + 1}')

    # Add spacing for the title
    # fig.suptitle('Language' + str(i+1), y=1.02)
    ax[2, 2].scatter([], [], label='Random testing',
                         color='red', marker='x')

    ax[2, 2].scatter([], [], label='Nominal active learning',
                         color='blue', marker='o')
    ax[2, 2].legend(loc='lower right')

    ax[2, 2].spines['top'].set_color('white')
    ax[2, 2].spines['right'].set_color('white')
    ax[2, 2].spines['bottom'].set_color('white')
    ax[2, 2].spines['left'].set_color('white')

    # Set the color of the x-axis and y-axis
    ax[2, 2].xaxis.label.set_color('white')
    ax[2, 2].yaxis.label.set_color('white')

    # Set the color of the tick labels
    ax[2, 2].tick_params(axis='x', colors='white')
    ax[2, 2].tick_params(axis='y', colors='white')

    # Show the plot
    plt.show()

def get_df_from_Table(df:pd.DataFrame, limitByTime:int, limitBySeq:int):
    column_name = df.columns
    for name in column_name:
        if 'Time' in str(name):
            x_axis_column = name
        if 'Total sequences asked before' in str(name):
            y_axis_column = name

    # print(f"x: {x_axis_column}, y: {y_axis_column}")
    df_selected =  df[[x_axis_column, y_axis_column]]

    new_df_filtered = df_selected
    if limitByTime is not None:
        new_df_filtered = df_selected[df_selected[x_axis_column] <= limitByTime]
    elif limitBySeq is not None:
        new_df_filtered = df_selected[df_selected[y_axis_column] <= limitBySeq]

    return new_df_filtered


def plot_line_graph_from_cumSum(df_list):
    num_plots = int(len(df_list) / 2)
    column_name = df_list[0].columns

    num_rows = 3
    num_cols = 3

    # Create a figure with subplots
    # fig, ax = plt.subplots(num_rows, num_cols, figsize=(8.27, 11.69), constrained_layout=True)  # A4 paper size
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(6, 6), constrained_layout=True)

    df_idx = 0
    for i in range(num_plots):
        # Calculate the row and column for the current subplot
        row = i // num_cols
        col = i % num_cols

        # Get the current DataFrame
        df1 = df_list[df_idx]
        df2 = df_list[df_idx + 1]
        # print(f"df1:{df_idx}, df2:{df_idx+1}")
        # print(f"row:{row}, col:{col}")

        df_idx += 2

        # plt.plot(df_line_graph[time_col], df_line_graph[y_col], marker='o', linestyle='-')
        ax[row, col].plot(df1[column_name[0]], df1[column_name[1]], label='Nominal active learning',
                             color='blue', linestyle='-') # , marker='o',
        ax[row, col].plot(df2[column_name[0]], df2[column_name[1]], label='Random testing',
                          color='red', linestyle='-')  # ,marker='x')

        ax[row, col].set_xlabel('Time (s)')

        ax[row, col].set_ylabel('Cumulative Adversarial examples')

        ax[row, col].set_title(f'Language {i + 1}')



    # Add spacing for the title
    # fig.suptitle('Language' + str(i+1), y=1.02)
    ax[2, 2].plot([], [], label='Random testing',
                     color='red', linestyle='-')

    ax[2, 2].plot([], [], label='Nominal active learning',
                     color='blue', linestyle='-')
    ax[2, 2].legend(loc='lower right')

    ax[2, 2].spines['top'].set_color('white')
    ax[2, 2].spines['right'].set_color('white')
    ax[2, 2].spines['bottom'].set_color('white')
    ax[2, 2].spines['left'].set_color('white')

    # Set the color of the x-axis and y-axis
    ax[2, 2].xaxis.label.set_color('white')
    ax[2, 2].yaxis.label.set_color('white')

    # Set the color of the tick labels
    ax[2, 2].tick_params(axis='x', colors='white')
    ax[2, 2].tick_params(axis='y', colors='white')

    # Show the plot
    plt.show()

def get_df_from_Table_for_freq(df: pd.DataFrame, limitByTime:int, limitBySeq:int):
    time_col = 'Time(s)'
    y_col = 'Adversarial Examples'
    df = df[~df[y_col].duplicated(keep='first')] # deletes any repetition of adversarial examples
    result_df = pd.DataFrame()
    result_df[time_col] = df[time_col]
    result_df[y_col] = 1

    if limitByTime:
        result_df = result_df[result_df[time_col] <= limitByTime]


    n = math.ceil(math.sqrt(result_df.shape[1])) # number of intervals
    time_range = result_df[time_col].max() - result_df[time_col].min()
    w = math.ceil(time_range/n) # width of intervals

    # print(result_df)
    s = 0
    x_col_for_df_per_sec, y_col_for_df_per_sec = [], []
    for interval_per_sec in range(0, int(math.ceil(result_df[time_col].max())), 1):
        # print(f"Looking for range: {interval_per_sec} : {interval_per_sec+1}")
        # print(f"xx:{result_df[result_df[time_col] >= interval_per_sec & (result_df[time_col] < interval_per_sec + 51 )][y_col]}")
        filtered = result_df[(result_df[time_col] >= interval_per_sec) & (result_df[time_col] <= interval_per_sec+1)]
        # print(f"xx:{filtered[y_col].sum()}")
        # sum_for_interval = result_df[result_df[time_col] >= interval_per_sec & (result_df[time_col] < interval_per_sec + 51 )][y_col].sum()
        # print(filtered)
        sum_for_interval = filtered[y_col].sum()
        s += sum_for_interval
        # print(f"sum:{sum_for_interval}, cumulative sum:{s}")
        x_col_for_df_per_sec.append(interval_per_sec)
        y_col_for_df_per_sec.append(s)

    df_per_sec = pd.DataFrame({time_col: x_col_for_df_per_sec,
                               y_col: y_col_for_df_per_sec})
    return df_per_sec


def get_graph_df(df: pd.DataFrame, interval_for_xaxis : int, limitByTime=None) -> pd.DataFrame:
    time_column_name = 'Time(s)'
    example_column_name = 'Adversarial Examples'
    result_2nd_column_name = 'No. of Examples found till time'
    result_df = pd.DataFrame(columns=[time_column_name, result_2nd_column_name])

    max_range = round(df[time_column_name].max()) + interval_for_xaxis - \
                (round(df[time_column_name].max()) % interval_for_xaxis)


    if limitByTime is not None:
        max_range = limitByTime - 1

    for time_interval in range(0, max_range + 1, interval_for_xaxis):
        examples_in_interval = df[df[time_column_name] <= time_interval][example_column_name].count()

        result_df = result_df._append({time_column_name: time_interval,
                                       result_2nd_column_name: examples_in_interval},
                                      ignore_index=True)
    return result_df

def main():
    parser = argparse.ArgumentParser(description='A Python script that to plot graphs.')
    parser.add_argument('--type', type=int, default=0, help='Specify an input value for language.')
    parser.add_argument('--file_suffix', type=str, default="", help='Add suffix for file name.')
    args = parser.parse_args()
    file_suffix = args.file_suffix
    type = args.type
    if file_suffix != "":
        file_suffix = "_" + file_suffix

    df_list = []
    for lang in range(1,9):
        if lang == 8:
            file_suffix = ""
        file1 = "lang" + str(lang) + "_detailedAdvExamples_nAL" + file_suffix + ".csv"
        file2 = "lang" + str(lang) + "_detailedAdvExamples_RS" + file_suffix + ".csv"
        print(f"Reading: file1:{file1}, file2:{file2}")
        if os.path.exists(file1):
            df1 = read_csv(file1)
        else:
            raise Exception(f"{file1} doesn't exist!")
        if os.path.exists(file2):
            df2 = read_csv(file2)
        else:
            raise Exception(f"{file2} doesn't exist!")

        if type == 0:
            df1_presentation = get_df_from_Table(df1, limitByTime=None, limitBySeq=1500)
            df2_presentation = get_df_from_Table(df2, limitByTime=None, limitBySeq=1500)
            df_list.append(df1_presentation)
            df_list.append(df2_presentation)

        elif type == 1:
            df1_presentation = get_df_from_Table_for_freq(df1, limitByTime=400, limitBySeq=None)
            df2_presentation = get_df_from_Table_for_freq(df2, limitByTime=400, limitBySeq=None)
            df_list.append(df1_presentation)
            df_list.append(df2_presentation)


    if type == 0:
        plot_dataframes_in_subplots(df_list)
    elif type == 1:
        plot_line_graph_from_cumSum(df_list)

if __name__ == "__main__":
    main()
