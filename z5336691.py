import json

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math
import re

studentid = os.path.basename(sys.modules[__name__].__file__)


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


# To compare IN、OUT、SAME
def compare_in_out(col1, col2):
    if col1 > col2:
        return "IN"
    elif col1 < col2:
        return "OUT"
    else:
        return "SAME"


# define function used in question2
def count_in_out(col, name1, name2, raw_df):
    #  count of “IN” for col for a unique “AustralianPort”
    out1 = df1[df1[col] == "IN"].groupby(["AustralianPort"]).count()
    out1 = out1.iloc[:, 0].reset_index()
    out1.rename(columns={'Month': name1}, inplace=True)

    # count of “OUT” for col for a unique “AustralianPort”
    out2 = df1[df1[col] == "OUT"].groupby(["AustralianPort"]).count()
    out2 = out2.iloc[:, 0].reset_index()
    out2.rename(columns={'Month': name2}, inplace=True)

    raw_df = pd.merge(raw_df, out1, on='AustralianPort', how='left')

    raw_df = pd.merge(raw_df, out2, on='AustralianPort', how='left')
    return raw_df


def get_average_by_country(col, df1, unique_month, name):
    # get the sum of col in every country and month
    df = pd.pivot_table(df1, values=col, index='Country', aggfunc='sum')
    df.rename(columns={col: name}, inplace=True)
    # get the mean value
    df[name] = round(df[name] / len(unique_month), 2)
    return df


def question_1(city_pairs):
    """
    :param city_pairs: the path for the routes dataset
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Load and read the data
    df1 = pd.read_csv(city_pairs)

    # create the new columns by comparing inbound and outbound values
    df1["passenger_in_out"] = df1.apply(lambda x: compare_in_out(x['Passengers_In'], x['Passengers_Out']), axis=1)
    df1["freight_in_out"] = df1.apply(lambda x: compare_in_out(x["Freight_In_(tonnes)"], x["Freight_Out_(tonnes)"]),
                                      axis=1)
    df1["mail_in_out"] = df1.apply(lambda x: compare_in_out(x["Mail_In_(tonnes)"], x["Mail_Out_(tonnes)"]), axis=1)
    # print(df1)
    #################################################

    log("QUESTION 1", output_df=df1[["AustralianPort", "ForeignPort",
                                     "passenger_in_out", "freight_in_out", "mail_in_out"]], other=df1.shape)
    return df1


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: dataframe df2
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################

    # filter out "SAME" rows and group by AustralianPort, summing the counts for IN/OUT of passengers, freight, and mail
    df2 = pd.DataFrame(columns=["AustralianPort"])
    df2["AustralianPort"] = df1["AustralianPort"].unique()
    df2 = count_in_out('passenger_in_out', 'PassengerInCount', 'PassengerOutCount', df2)
    df2 = count_in_out('freight_in_out', 'FreightInCount', 'FreightOutCount', df2)
    df2 = count_in_out('mail_in_out', 'MailInCount', 'MailOutCount', df2)
    df2 = df2.fillna(0)
    # sorted in descending order by "PassengerInCount”(highest to lowest)
    df2 = df2.sort_values(by="PassengerInCount", ascending=False)
    #################################################

    log("QUESTION 2", output_df=df2, other=df2.shape)
    return df2


def question_3(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    unique_month = df1['Month'].unique()
    df_list = [
        get_average_by_country('Passengers_In', df1, unique_month, 'Passengers_in_average'),
        get_average_by_country('Passengers_Out', df1, unique_month, 'Passengers_out_average'),
        get_average_by_country('Freight_In_(tonnes)', df1, unique_month, 'Freight_in_average'),
        get_average_by_country('Freight_Out_(tonnes)', df1, unique_month, 'Freight_out_average'),
        get_average_by_country('Mail_In_(tonnes)', df1, unique_month, 'Mail_in_average'),
        get_average_by_country('Mail_Out_(tonnes)', df1, unique_month, 'Mail_out_average')
    ]
    df3 = pd.concat(df_list, axis=1)
    df3 = df3.groupby('Country').mean().reset_index()
    df3 = df3.sort_values(by="Passengers_in_average", ascending=True)
    #################################################

    log("QUESTION 3", output_df=df3, other=df3.shape)
    return df3


def question_4(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # create a new dataframe with only the necessary columns
    # Your code goes here ...
    # get unique country
    # Filter the city_pairs dataframe to include only rows where Passengers_Out is greater than 0
    filtered = df1[df1['Passengers_Out'] > 0]

    # Group the filtered dataframe by AustralianPort, Country, and Month and aggregate the count of ForeignPort
    grouped = filtered.groupby(['AustralianPort', 'Country', 'Month']).agg(Count=('ForeignPort', 'count')).reset_index()

    # Filter the grouped dataframe to include only rows where the Count column is greater than 1
    filtered_grouped = grouped[grouped['Count'] > 1]

    # Group the filtered_grouped dataframe by Country and aggregate the sum of Count
    agg_df = filtered_grouped.groupby('Country').agg(Unique_ForeignPort_Count=('Count', 'count')).reset_index()

    # Sort the aggregated dataframe by Count in descending order and return the first 5 rows
    df4 = agg_df.sort_values(by='Unique_ForeignPort_Count', ascending=False).head(5)
    #################################################
    log("QUESTION 4", output_df=df4, other=df4.shape)
    return df4


def question_5(seats):
    """
    :param seats : the path to dataset
    :return: df5
            Data Type: dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # load the seats dataset
    df5 = pd.read_csv(seats)

    # create new columns "Source_City" and "Destination_City" based on the "In_Out" column
    df5["Source_City"] = np.where(df5["In_Out"] == "I", df5["International_City"], df5["Australian_City"])
    df5["Destination_City"] = np.where(df5["In_Out"] == "I", df5["Australian_City"], df5["International_City"])
    # print(df5)
    #################################################

    log("QUESTION 5", output_df=df5, other=df5.shape)
    return df5


def question_6(df5):
    """
    :param df5: the dataframe created in question 5
    :return: df6
    """
    #################################################
    df6 = df5.drop(['In_Out', 'Port_Country', 'Port_Region', 'Service_Country', 'Service_Region', 'Year'], axis=1)
    df6 = df6.groupby(by=['Source_City', 'Destination_City', 'Month_num']).agg({'Airline': lambda x: len(x.unique()),
                                                                                'All_Flights': 'sum',
                                                                                'Max_Seats': 'mean'})
    df6 = df6.reset_index()
    df6 = df6.sort_values(by=['Airline', 'All_Flights'], ascending=[False, False])

    """
    Firstly, we remove some useless columns from the data obtained from Question5 and cluster according to [ 
    'Source_City','Destination_City','Month_num'] to get the number of airlines and total number of flights for each 
    route in month. As shown in this question6 table. We rank them from largest to smallest by the number of airlines 
    and max_seat. In this way, when airlines see this table, they can see the corresponding hot routes, 
    the total number of flights, and the number of airlines under each month. Can clearly understand the current time 
    under the route operation situation. Therefore, it is conducive for the airlines to conduct targeted research. 
    For example, it can be seen from the figure that in September, the most significant number of carrier airlines 
    flew from Auckland to Sydney, and the number of flights was also the largest. Therefore, new airlines must face 
    relatively strong competitiveness when investing in this route. In conclusion, it provides airlines with insights 
    to help them decide which routes to invest in. To that end, one possible addition to the analysis is to include 
    data on passenger demand for each route. By including this additional data, airlines can better understand the 
    competition they may face on a given route and the potential demand for their services. This information helps 
    airlines make more informed decisions about route planning and scheduling and helps them identify opportunities 
    to expand their services in specific markets.
    """
    #################################################

    log("QUESTION 6", output_df=df6, other=df6.shape)
    return df6


def question_7(seats, city_pairs):
    """
    :param seats: the path to dataset
    :param city_pairs : the path to dataset
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Load the datasets

    df1 = question_1(city_pairs)
    df5 = question_5(seats)

    df1_1 = df1.iloc[49668:]
    df5_1 = df5.copy()
    df5_1.rename(columns={'Australian_City': 'AustralianPort', 'International_City': 'ForeignPort'}, inplace=True)
    df5_1 = df5_1.drop_duplicates(['Month', 'AustralianPort', 'ForeignPort'])
    df_merge = pd.merge(df1_1, df5_1, on=['Month', 'AustralianPort', 'ForeignPort'], how='left')
    df_merge = df_merge[
        ['Month', 'AustralianPort', 'ForeignPort', 'Passengers_In', 'Passengers_Out', 'Port_Region', 'Max_Seats']]

    d = df_merge.groupby(by=['Month', 'Port_Region']).mean().reset_index()
    d['Month'] = pd.to_datetime(d['Month'], format='%b-%y')
    d['Month'] = d['Month'].dt.strftime('%Y-%m')
    d = d.sort_values('Month')

    plt.figure(figsize=(20, 10))
    x = d['Month'].unique()
    Port_Region = ['S America', 'N America']
    for i in Port_Region:
        d1 = d[d['Port_Region'] == i]
        plt.scatter(d1['Month'].unique(), d1['Passengers_In'] / d1['Max_Seats'],
                    label='seat utilisation_In of {}'.format(i))
        plt.scatter(d1['Month'].unique(), d1['Passengers_Out'] / d1['Max_Seats'],
                    label='seat utilisation_Out of {}'.format(i))
    plt.xlabel('time series', size=16)
    plt.xticks(rotation=45, size=16)
    plt.yticks(size=16)
    plt.xticks(x[::10])
    plt.ylabel('variable of seat utilisation', size=16)
    plt.legend(fontsize=16)

    """First of all, we combine the two tables according to 'Month','AustralianPort' and 'ForeignPort' to obtain the 
    number of passengers corresponding to each route. The maximum seat number is Port_Region. We selected two 
    port_regions, i.e., ['S America','N America'], to plot the average seat utilization of the two regions in each 
    month's flight volume. As shown in the picture. 
    
    This figure shows that Passengers_In and Passengers_Out 
    correspond to the same Port_Region with the same seat utilization ratio. Through this picture, airlines can see 
    the seat utilization situation and the difference in each time node corresponding to the region. 
    In the long run, airlines can make more informed decisions regarding capacity planning and pricing strategies. 
    For example, airlines can adjust the number of flights and the seating capacity for specific routes based on the seat 
    utilization rate and the corresponding region. By doing so, they can optimize their revenue while meeting the 
    demand of passengers. Moreover, declining seat utilization rates in both S America and N America after 2020 
    suggest that the COVID-19 pandemic has significantly impacted the airline industry. As the world recovers from 
    the pandemic, airlines can use the insights provided by this visualization to develop strategies to recover from 
    the crisis and adapt to the new market conditions. Overall, this visualization provides a valuable tool for 
    airlines to understand the dynamics of the aviation market and make data-driven decisions to improve their 
    operational efficiency and profitability.
    """
    #################################################

    plt.savefig("{}-Q7.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("city_pairs.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df1.copy(True))
    df4 = question_4(df1.copy(True))
    df5 = question_5("seats.csv")
    df6 = question_6(df5.copy(True))
    question_7("seats.csv", "city_pairs.csv")
