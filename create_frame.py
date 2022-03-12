import glob
import json

import pandas as pd

import calc_measures_sol as measures_chiara
import knn_conv_costraint_NEW_con_norm as knn_cov

path = r'C:\Users\Nicolò\Desktop\Tesi\res2  - rewrinting'  # use your path
all_files = glob.glob(path + "/*.csv")
result_csv = pd.DataFrame(columns=['query',
                                   'path',
                                   'card_true_tot_Q',
                                   'card_true_sa_Q',
                                   'card_true_tot_newQ',
                                   'card_true_sa_newQ',
                                   'proximity_distinct',
                                   'proximity_qcut',
                                   'relaxation_degree',
                                   'disparity_index',
                                   'fairness_index',
                                   'distinct_average_time_preprocessing',
                                   'distinct_average_time_pruning',
                                   'distinct_average_time_algo',
                                   'distinct_mean_summed_time',
                                   'qcut_average_time_preprocessing',
                                   'qcut_average_time_pruning',
                                   'qcut_average_time_algo',
                                   'qcut_mean_summed_time',
                                   'average_time_preprocessing',
                                   'average_time_pruning',
                                   'average_time_algo',
                                   'mean_summed_time'
                                   ])
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    card_true_sa_Q = list(map(int, df['card_true_sa_Q'].iloc[0].strip('[]').split(',')))
    card_true_sa_newQ = list(map(int, df['card_true_sa_newQ'].iloc[0].strip('[]').split(',')))
    qcut_first_row = df.loc[df['preprocessing'] == 'qcut'].iloc[0]
    # le cardinalità devono essere differenziate a seconda del preprocessing ?
    proximity_distinct = measures_chiara.proximity(json.loads(df['output_prep'].iloc[0]), df['solution'].iloc[0],
                                                   df['n_bin'].iloc[0], '')
    proximity_qcut = measures_chiara.proximity(json.loads(qcut_first_row['output_prep']), qcut_first_row['solution'],
                                               qcut_first_row['n_bin'], '')
    relaxation_degree = knn_cov.get_relaxation_degree(df['card_true_tot_Q'].iloc[0],
                                                      df['card_true_tot_newQ'].iloc[0])
    disparity_index = knn_cov.measure_DispInd(card_true_sa_newQ,
                                              int(df['card_true_tot_newQ'].iloc[0]))
    fairness_index = knn_cov.measure_FairInd(df['card_true_tot_Q'].iloc[0],
                                             df['card_true_tot_newQ'].iloc[0],
                                             card_true_sa_Q,
                                             card_true_sa_newQ)
    distinct_dataframe = df.loc[df['preprocessing'] == 'distinct']
    summed_times_distinct = distinct_dataframe.loc[:,
                            ['time_preprocessing', 'time_pruning', 'time_algo']].sum(axis=1).mean()
    qcut_dataframe = df.loc[df['preprocessing'] == 'qcut']
    summed_times_qcut = qcut_dataframe.loc[:,
                        ['time_preprocessing', 'time_pruning', 'time_algo']].sum(axis=1).mean()
    summed_times = df.loc[:, ['time_preprocessing', 'time_pruning', 'time_algo']].sum(axis=1).mean()
    s = pd.Series(
        [df['query'].iloc[0], filename, df['card_true_tot_Q'].iloc[0], card_true_sa_Q, df['card_true_tot_newQ'].iloc[0],
         card_true_sa_newQ, proximity_distinct, proximity_qcut, relaxation_degree, disparity_index, fairness_index,
         distinct_dataframe['time_preprocessing'].mean(),
         distinct_dataframe['time_pruning'].mean(), distinct_dataframe['time_algo'].mean(),
         summed_times_distinct,
         qcut_dataframe['time_preprocessing'].mean(), qcut_dataframe['time_pruning'].mean(),
         qcut_dataframe['time_algo'].mean(), summed_times_qcut,
         df['time_preprocessing'].mean(), df['time_pruning'].mean(), df['time_algo'].mean(), summed_times],
        index=result_csv.columns)
    result_csv = result_csv.append(s, ignore_index=True)

result_csv.to_csv(r'C:\Users\Nicolò\Desktop\Tesi\res2  - rewrinting\test_result_1.csv', index=False)
writer = pd.ExcelWriter(r'C:\Users\Nicolò\Desktop\Tesi\res2 - xls_file\test_result_1.xlsx')
result_csv.to_excel(writer)
writer.save()
