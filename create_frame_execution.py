import glob
import json

import numpy as np
import pandas as pd

import calc_measures_sol as measures_chiara
import knn_conv_costraint_NEW_con_norm as knn_cov

path = r'C:\Users\Nicolò\Desktop\Tesi\res2 - execution'  # use your path
all_files = glob.glob(path + "/*.csv")

result_csv = pd.DataFrame(columns=['query',
                                   'path',
                                   'Coverage Constraint',
                                   'card_true_tot_Q',
                                   'card_true_sa_Q',
                                   'card_true_tot_newQ',
                                   'card_true_sa_newQ',
                                   'relaxation_degree',
                                   'disparity_index',
                                   'fairness_index',
                                   'average_time_read_table',
                                   'average_time_norm_data_and_comp_q',
                                   'average_time_norm_data',
                                   'average_time_q_comp',
                                   'average_time_norm_point',
                                   'average_time_tree_exec',
                                   'average_time_execution'])

Qind_result_csv = pd.DataFrame(columns=['Qind',
                                        'path',
                                        'Coverage constraint',
                                        'card_true_tot_Q',
                                        'card_true_sa_Q',
                                        'card_true_tot_Qind',
                                        'card_true_sa_Qind',
                                        'relaxation_degree_Qind',
                                        'disparity_index_Qind',
                                        'fairness_index_Qind',
                                        'proximity_Qind',
                                        'average_time_exec_Qind'
                                        ])


def average_a_lst_in_df (series):
    list1 = []
    list2 = []
    for index, value in series.items():
        list1.append(float(value[0]))
        list2.append(float(value[1]))
    res = [np.mean(list1), np.mean(list2)]
    return res


for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df['time_norm_data_and_comp_q'] = df['time_norm_data_and_comp_q'].map(lambda x: x.strip('[]'))
    df['time_norm_data'] = df['time_norm_data'].map(lambda x: x.strip('[]'))
    df['time_tree_exec'] = df['time_tree_exec'].map(lambda x: x.strip('[]'))
    if (',' in df['time_norm_data_and_comp_q'].iloc[0]):
        df['time_norm_data_and_comp_q'] = df['time_norm_data_and_comp_q'].map(lambda x: x.split(','))
        list_mean_per_norm_and_comp_q = average_a_lst_in_df(df['time_norm_data_and_comp_q'])
        df['time_norm_data'] = df['time_norm_data'].map(lambda x: x.split(','))
        list_mean_per_time_norm_data = average_a_lst_in_df(df['time_norm_data'])
        df['time_tree_exec'] = df['time_tree_exec'].map(lambda x: x.split(','))
        list_mean_per_time_tree_exec = average_a_lst_in_df(df['time_tree_exec'])
        s = pd.Series([df['query'].iloc[0], filename, df['CC'].iloc[0], df['card_true_tot_Q'].iloc[0],
                       df['card_true_sa_Q'].iloc[0],
                       df['card_tot_res'].iloc[0], df['card_AS_res'].iloc[0],
                       df['relaxation_degree_res'].iloc[0], df['disparity_index_res'].iloc[0],
                       df['fairness_index_res'].iloc[0], df['time_read_table'].mean(),
                       list_mean_per_norm_and_comp_q,
                       list_mean_per_time_norm_data, df['time_q_comp'].mean(), df['time_norm_point'].mean(),
                       list_mean_per_time_tree_exec, df['time_execution'].mean()],
                      index=result_csv.columns)
        Qind_s = pd.Series([df['Q_ind'].iloc[0], filename, df['CC'].iloc[0], df['card_true_tot_Q'].iloc[0],
                            df['card_true_sa_Q'].iloc[0],
                            df['card_true_tot_Qind'].iloc[0], df['card_true_sa_Qind'].iloc[0],
                            df['relaxation_degree_Qind'].iloc[0],
                            df['fairness_index_Qind'].iloc[0], df['disparity_index_Qind'].iloc[0],
                            df['proximity_Qind'].iloc[0], df['time_exec_Qind'].mean()], index=Qind_result_csv.columns)
        Qind_result_csv = Qind_result_csv.append(Qind_s, ignore_index=True)
        result_csv = result_csv.append(s, ignore_index=True)
        continue
    df['time_norm_data_and_comp_q'] = df['time_norm_data_and_comp_q'].map(float)
    df['time_norm_data'] = df['time_norm_data'].map(float)
    df['time_tree_exec'] = df['time_tree_exec'].map(float)
    s = pd.Series(
        [df['query'].iloc[0], filename, df['CC'].iloc[0], df['card_true_tot_Q'].iloc[0], df['card_true_sa_Q'].iloc[0],
         df['card_tot_res'].iloc[0], df['card_AS_res'].iloc[0],
         df['relaxation_degree_res'].iloc[0], df['disparity_index_res'].iloc[0],
         df['fairness_index_res'].iloc[0], df['time_read_table'].mean(),
         df['time_norm_data_and_comp_q'].mean(),
         df['time_norm_data'].mean(), df['time_q_comp'].mean(), df['time_norm_point'].mean(),
         df['time_tree_exec'].mean(), df['time_execution'].mean()],
        index=result_csv.columns)
    result_csv = result_csv.append(s, ignore_index=True)
    Qind_s = pd.Series(
        [df['Q_ind'].iloc[0], filename, df['CC'].iloc[0], df['card_true_tot_Q'].iloc[0], df['card_true_sa_Q'].iloc[0],
         df['card_true_tot_Qind'].iloc[0], df['card_true_sa_Qind'].iloc[0],
         df['relaxation_degree_Qind'].iloc[0],
         df['fairness_index_Qind'].iloc[0], df['disparity_index_Qind'].iloc[0],
         df['proximity_Qind'].iloc[0], df['time_exec_Qind'].mean()], index=Qind_result_csv.columns)
    Qind_result_csv = Qind_result_csv.append(Qind_s, ignore_index=True)

result_csv.to_csv(r'C:\Users\Nicolò\Desktop\Tesi\res2 - execution\test_result_exec_1.csv', index=False)
Qind_result_csv.to_csv(r'C:\Users\Nicolò\Desktop\Tesi\res2 - execution\test_Qind_exec_1.csv', index=False)
writer = pd.ExcelWriter(r'C:\Users\Nicolò\Desktop\Tesi\res2 - xls_file\test_result_exec_1.xlsx')
result_csv.to_excel(writer)
writer_1 = pd.ExcelWriter(r'C:\Users\Nicolò\Desktop\Tesi\res2 - xls_file\test_Qind_exec_1.xlsx')
Qind_result_csv.to_excel(writer_1)
writer.save()
writer_1.save()
