import glob
import os

import pandas as pd
from openpyxl.workbook import Workbook

import knn_conv_costraint_NEW_con_norm

path = r'C:\Users\Nicolò\Desktop\Tesi\res2  - rewrinting'  # use your path
all_files = glob.glob(path + "/*.csv")
result_csv = pd.DataFrame(columns=['query',
                                   'path',
                                   'relaxation_degree',
                                   'disparity_index',
                                   'fairness_index',
                                   'mean_time',
                                   ])
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    card_true_sa_Q = list(map(int, df['card_true_sa_Q'].iloc[0].strip('[]').split(',')))
    card_true_sa_newQ = list(map(int, df['card_true_sa_newQ'].iloc[0].strip('[]').split(',')))

    relaxation_degree = knn_conv_costraint_NEW_con_norm.get_relaxation_degree(df['card_true_tot_Q'].iloc[0],
                                                                              df['card_true_tot_newQ'].iloc[0])
    disparity_index = knn_conv_costraint_NEW_con_norm.measure_DispInd(card_true_sa_newQ,
                                                                      int(df['card_true_tot_newQ'].iloc[0]))
    fairness_index = knn_conv_costraint_NEW_con_norm.measure_FairInd(df['card_true_tot_Q'].iloc[0],
                                                                     df['card_true_tot_newQ'].iloc[0],
                                                                     card_true_sa_Q,
                                                                     card_true_sa_newQ)
    df['summed_times'] = df.loc[:, ['time_preprocessing', 'time_pruning', 'time_algo']].sum(axis=1)
    mean_time = df['summed_times'].mean()
    s = pd.Series([df['query'].iloc[0], filename, relaxation_degree, disparity_index, fairness_index, mean_time],
                  index=result_csv.columns)
    result_csv = result_csv.append(s, ignore_index=True)

result_csv.to_csv(r'C:\Users\Nicolò\Desktop\Tesi\res2  - rewrinting\test_result_1.csv', index=False)
writer = pd.ExcelWriter(r'C:\Users\Nicolò\Desktop\Tesi\res2 - xls_file\test_result_1.xlsx')
result_csv.to_excel(writer)
writer.save()
# frame = pd.concat(li, axis=0, ignore_index=True)
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# print(knn_conv_constraint_norm.measure_DispInd(1,0))
# TODO misure per i file di rewrinting più la somma dei tempi per creare il tempo totale
# TODO calcolare il tempo medio per ogni query quinfi creare un nuovo data frame
