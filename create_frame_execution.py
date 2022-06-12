import glob

import numpy as np
import pandas as pd

path = r'C:\Users\Nicolò\Desktop\Tesi\res_tesi\exec_csv'  # use your path
all_files = glob.glob(path + "/*.csv")

result_csv = pd.DataFrame(columns=[
    'id',
    'query',
    'Coverage Constraint',
    'card_true_tot_Q',
    'card_true_sa_Q',
    'card_true_tot_newQ',
    'card_true_sa_newQ',
    'relaxation_degree',
    'disparity_index',
    'fairness_index',
    'average_time_init_numero_cond',
    'average_time_for_computing_4groups',
    'average_time_for_computing_df',
    'average_time_for_computing_df_cc',
    'average_time_query_as',
    'average_time_for_computing_2df_res',
    'average_time_op_varie',
    'average_time_if',
    'average_time_res_no_duplicates',
    'average_time_finalres',
    'average_time_read_table',
    'average_time_list_col',
    'average_time_norm_data',
    'average_time_tree_exec',
    'average_time_execution'])

Qind_result_csv = pd.DataFrame(columns=[
    'id', 'Qind',
    'Coverage Constraint',
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
    id = int(filename[filename.find('Q'):].strip('Q').strip('_norm.csv'))
    if ',' in str(df['time_norm_data'].iloc[0]):
        list_mean_per_time_res_no_duplicates = average_a_lst_in_df(
            df['time_res_no_duplicates'].map(lambda x: x.strip('[]')).map(lambda x: x.split(',')))
        list_mean_per_time_norm_data = average_a_lst_in_df(
            df['time_norm_data'].map(lambda x: x.strip('[]')).map(lambda x: x.split(',')))
        list_mean_per_time_tree_exec = average_a_lst_in_df(
            df['time_tree_exec'].map(lambda x: x.strip('[]')).map(lambda x: x.split(',')))

        s = pd.Series([id, df['query'].iloc[0], df['CC'].iloc[0], df['card_true_tot_Q'].iloc[0],
                       df['card_true_sa_Q'].iloc[0],
                       df['card_tot_res'].iloc[0], df['card_AS_res'].iloc[0],
                       df['relaxation_degree_res'].iloc[0], df['disparity_index_res'].iloc[0],
                       df['fairness_index_res'].iloc[0], df['time_init_numero_cond'].mean(), 0,
                       df['time_for_computing_df'].mean(), 0, df['time_query_as'].mean(), 0, 0, 0,
                       list_mean_per_time_res_no_duplicates,
                       df['time_finalres'].mean(), df['time_read_table'].mean(), df['time_list_col'].mean(),
                       list_mean_per_time_norm_data,
                       list_mean_per_time_tree_exec, df['time_execution'].mean()],
                      index=result_csv.columns)

        Qind_s = pd.Series([id, df['Q_ind'].iloc[0], df['CC'].iloc[0], df['card_true_tot_Q'].iloc[0],
                            df['card_true_sa_Q'].iloc[0],
                            df['card_true_tot_Qind'].iloc[0], df['card_true_sa_Qind'].iloc[0],
                            df['relaxation_degree_Qind'].iloc[0],
                            df['fairness_index_Qind'].iloc[0], df['disparity_index_Qind'].iloc[0],
                            df['proximity_Qind'].iloc[0], df['time_exec_Qind'].mean()],
                           index=Qind_result_csv.columns)
        Qind_result_csv = Qind_result_csv.append(Qind_s, ignore_index=True)
        result_csv = result_csv.append(s, ignore_index=True)
        continue

    df['time_norm_data'] = df['time_norm_data'].map(float)
    df['time_tree_exec'] = df['time_tree_exec'].map(float)

    s = pd.Series(
        [id, df['query'].iloc[0], df['CC'].iloc[0], df['card_true_tot_Q'].iloc[0],
         df['card_true_sa_Q'].iloc[0],
         df['card_tot_res'].iloc[0], df['card_AS_res'].iloc[0],
         df['relaxation_degree_res'].iloc[0], df['disparity_index_res'].iloc[0],
         df['fairness_index_res'].iloc[0], df['time_init_numero_cond'].mean(), df['time_for_computing_4groups'].mean(),
         0, df['time_for_computing_df_cc'].mean(), 0, df['time_for_computing_2df_res'].mean(),
         df['time_op_varie'].mean(), df['time_if'].mean(), df['time_res_no_duplicates'].mean(),
         df['time_finalres'].mean(), df['time_read_table'].mean(), df['time_list_col'].mean(),
         df['time_norm_data'].mean(),
         df['time_tree_exec'].mean(), df['time_execution'].mean()],
        index=result_csv.columns)
    result_csv = result_csv.append(s, ignore_index=True)

    Qind_s = pd.Series(
        [id, df['Q_ind'].iloc[0], df['CC'].iloc[0], df['card_true_tot_Q'].iloc[0],
         df['card_true_sa_Q'].iloc[0],
         df['card_true_tot_Qind'].iloc[0], df['card_true_sa_Qind'].iloc[0],
         df['relaxation_degree_Qind'].iloc[0],
         df['disparity_index_Qind'].iloc[0], df['fairness_index_Qind'].iloc[0],
         df['proximity_Qind'].iloc[0], df['time_exec_Qind'].mean()], index=Qind_result_csv.columns)
    Qind_result_csv = Qind_result_csv.append(Qind_s, ignore_index=True)

result_csv.sort_values(by="id", inplace=True)
Qind_result_csv.sort_values(by="id", inplace=True)

result_csv.to_csv(r'C:\Users\Nicolò\Desktop\Tesi\res_tesi\df_risultati\result_exec_1.csv', index=False)
Qind_result_csv.to_csv(r'C:\Users\Nicolò\Desktop\Tesi\res_tesi\df_risultati\Qind_exec_1.csv', index=False)
writer = pd.ExcelWriter(r'C:\Users\Nicolò\Desktop\Tesi\res_tesi\df_risultati\result_exec_1.xlsx')
result_csv.to_excel(writer)
writer_1 = pd.ExcelWriter(r'C:\Users\Nicolò\Desktop\Tesi\res_tesi\df_risultati\Qind_exec_1.xlsx')
Qind_result_csv.to_excel(writer_1)
writer.save()
writer_1.save()
