import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import numpy as np
# import sys
import argparse

# lang = 3
def read_csv(file:str) -> pd.DataFrame:
    if not os.path.exists(file):
        raise Exception("file doesn't exist")
    return pd.read_csv(file)

def plot_overview_graph_from_df(df1:pd.DataFrame, df2: pd.DataFrame, roundOffxaxis=True):
    # Create a subplot with two plots sharing the same X-axis
    if roundOffxaxis==False:
        fig, ax = plt.subplots(figsize=(15, 6))
    else:
        fig, ax = plt.subplots()

    column_name = df1.columns

    if not df1.columns.equals(df2.columns):
        raise Exception(f"The columns name of the two data frame are different. df1: {df1.columns} and df2: {df2.columns}")

    # Plot the first dataset (df1) with a blue line
    # ax.plot(df1[column_name[0]], df1[column_name[1]], label='Number of adversarial examples using  extraction of automata', color='blue',
    #         marker='o', linestyle=None)
    ax.scatter(df1[column_name[0]], df1[column_name[1]], label='Active learning approach', color='blue',
            marker='o')

    # Plot the second dataset (df2) with a red line
    # ax.plot(df2[column_name[0]], df2[column_name[1]], label='Number of adversarial examples using  using random sampling', color='red',
    #         marker='x', linestyle=None)
    ax.scatter(df2[column_name[0]], df2[column_name[1]], label='Random testing', color='red',
            marker='x')

    # Set labels and title
    ax.set_xlabel(column_name[0])
    if 'before' in (column_name[1]):
        ax.set_ylabel("Total number of sequences checked before", wrap=True)
    else:
        ax.set_ylabel(column_name[1])
    ax.set_title('Active learning approach vs. Random testing')
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


def get_df_from_presentationTable(df:pd.DataFrame, limitByTime:int, limitBySeq:int):
    column_name = df.columns
    for name in column_name:
        if 'Time' in str(name):
            x_axis_column = name
        if 'Total sequences asked before' in str(name):
            y_axis_column = name

    print(f"x: {x_axis_column}, y: {y_axis_column}")
    df_selected =  df[[x_axis_column, y_axis_column]]

    new_df_filtered = df_selected
    if limitByTime is not None:
        new_df_filtered = df_selected[df_selected[x_axis_column] <= limitByTime]
    elif limitBySeq is not None:
        new_df_filtered = df_selected[df_selected[y_axis_column] <= limitBySeq]

    return new_df_filtered

def get_cumulative_df_per_actual_time(df: pd.DataFrame):
    time_column_name = 'Time(s)'
    example_column_name = 'Adversarial Examples'
    result_2nd_column_name = 'No. of Examples found till time'
    # result_df = pd.DataFrame(columns=[time_column_name, result_2nd_column_name])

    df[example_column_name] = 1
    result_df = pd.DataFrame()
    result_df[time_column_name] = df[time_column_name]
    result_df[result_2nd_column_name] = df[example_column_name].cumsum()

    # print(df)
    return result_df


def get_graph_df(df: pd.DataFrame, interval_for_xaxis : int, limitByTime=None) -> pd.DataFrame:
    time_column_name = 'Time(s)'
    example_column_name = 'Adversarial Examples'
    result_2nd_column_name = 'No. of Examples found till time'
    result_df = pd.DataFrame(columns=[time_column_name, result_2nd_column_name])

    max_range = round(df[time_column_name].max()) + interval_for_xaxis - \
                (round(df[time_column_name].max()) % interval_for_xaxis)


    if limitByTime is not None:
        max_range = limitByTime - 1
    # Iterate through time intervals
    print(max_range)
    for time_interval in range(0, max_range + 1, interval_for_xaxis):
        # Count the number of examples within the current time interval
        examples_in_interval = df[df[time_column_name] <= time_interval][example_column_name].count()

        # Update the cumulative count
        # cumulative_count += examples_in_interval

        # Append the results to the new DataFrame
        result_df = result_df._append({time_column_name: time_interval,
                                       result_2nd_column_name: examples_in_interval},
                                      ignore_index=True)
    # print(result_df)
    return result_df

def main():
    parser = argparse.ArgumentParser(description='A Python script that to plot graphs.')
    parser.add_argument('--lang', type=int, help='Specify an input value for language.')
    parser.add_argument('--file_suffix', type=str, default="", help='Add suffix for file name.')
    args = parser.parse_args()
    global lang
    lang = args.lang
    file_suffix = args.file_suffix

    print(f"Language selected is:{str(lang)}")
    filelist = ["lang" + str(lang) + "_adversarial_list.csv",
                "lang" + str(lang) + "_adversarial_list_rSampling.csv"]

    df1 = read_csv(filelist[0])
    df2 = read_csv(filelist[1])
    interval_for_xaxis = 10
    #actual_plot
    # df1_cumulative_per_actual_time = get_cumulative_df_per_actual_time(df=df1)
    # df2_cumulative_per_actual_time = get_cumulative_df_per_actual_time(df=df2)

    # print(df1_cumulative_per_actual_time)
    # plot_overview_graph_from_df(df1_cumulative_per_actual_time, df2_cumulative_per_actual_time, roundOffxaxis=False)

    #gives overview
    # df1_ready_to_plot = get_graph_df(df=df1, interval_for_xaxis=interval_for_xaxis, limitByTime=400)
    # df2_ready_to_plot = get_graph_df(df=df2, interval_for_xaxis=interval_for_xaxis, limitByTime=400)
    # plot_overview_graph_from_df(df1_ready_to_plot, df2_ready_to_plot)


    file_suffix = "_"

    filelist = ["lang" + str(lang) + "_presentationTable" + file_suffix + ".csv",
                "lang" + str(lang) + "_presentationTable_rSampling" + file_suffix + ".csv"]

    df1 = read_csv(filelist[0])
    df2 = read_csv(filelist[1])
    df1_presentation = get_df_from_presentationTable(df1, limitByTime=None, limitBySeq=2000)
    df2_presentation = get_df_from_presentationTable(df2, limitByTime=None, limitBySeq=2000)
    
    plot_overview_graph_from_df(df1_presentation, df2_presentation)



if __name__ == "__main__":
    main()
