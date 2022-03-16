import glob
import json

import pandas as pd

import calc_measures_sol as measures_chiara
import knn_conv_costraint_NEW_con_norm as knn_cov

path = r'C:\Users\Nicolò\Desktop\Tesi\res2  - rewrinting'  # use your path
all_files = glob.glob(path + "/*.csv")
pd.set_option("display.max_rows", None, "display.max_columns", None)
result_csv = pd.DataFrame(columns=[
    'id',
    'query',
    'path',
    'Coverage costraint',
    'card_true_tot_Q',
    'card_true_sa_Q',
    'card_true_tot_newQ',
    'card_true_sa_newQ',
    'proximity_qcut',
    'relaxation_degree',
    'disparity_index',
    'fairness_index',
    'qcut_average_time_preprocessing',
    'qcut_average_time_pruning',
    'qcut_average_time_algo',
    'qcut_mean_summed_time'
])

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    qcut_first_row = df.loc[df['preprocessing'] == 'qcut'].iloc[0]
    card_true_sa_Q = list(map(int, qcut_first_row['card_true_sa_Q'].strip('[]').split(',')))
    card_true_sa_newQ = list(map(int, qcut_first_row['card_true_sa_newQ'].strip('[]').split(',')))
    proximity_qcut = measures_chiara.proximity(json.loads(qcut_first_row['output_prep']), qcut_first_row['solution'],
                                               qcut_first_row['n_bin'], '')
    relaxation_degree = knn_cov.get_relaxation_degree(qcut_first_row['card_true_tot_Q'],
                                                      qcut_first_row['card_true_tot_newQ'])
    disparity_index = knn_cov.measure_DispInd(card_true_sa_newQ,
                                              int(qcut_first_row['card_true_tot_newQ']))
    fairness_index = knn_cov.measure_FairInd(qcut_first_row['card_true_tot_Q'],
                                             qcut_first_row['card_true_tot_newQ'],
                                             card_true_sa_Q,
                                             card_true_sa_newQ)
    qcut_dataframe = df.loc[df['preprocessing'] == 'qcut']
    summed_times_qcut = qcut_dataframe.loc[:,
                        ['time_preprocessing', 'time_pruning', 'time_algo']].sum(axis=1).mean()
    s = pd.Series(
        [int(filename[65:].strip('.csv').strip('Q')), df['query'].iloc[0], filename, df['CC'].iloc[0],
         qcut_first_row['card_true_tot_Q'], card_true_sa_Q,
         qcut_first_row['card_true_tot_newQ'],
         card_true_sa_newQ, proximity_qcut, relaxation_degree, disparity_index, fairness_index,
         qcut_dataframe['time_preprocessing'].mean(), qcut_dataframe['time_pruning'].mean(),
         qcut_dataframe['time_algo'].mean(), summed_times_qcut],
        index=result_csv.columns)
    result_csv = result_csv.append(s, ignore_index=True)

result_csv.sort_values(by="id", inplace=True)

result_csv.to_csv(r'C:\Users\Nicolò\Desktop\Tesi\result_experiment\test_result_1.csv', index=False)
writer = pd.ExcelWriter(r'C:\Users\Nicolò\Desktop\Tesi\res2 - xls_file\test_result_1.xlsx')
result_csv.to_excel(writer)
writer.save()
