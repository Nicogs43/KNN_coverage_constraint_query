import glob

import pandas as pd

path = r'C:\Users\Nicolò\Desktop\Tesi\res_tesi\qcut_query_csv'  # use your path
all_files = glob.glob(path + "/*.csv")
pd.set_option("display.max_rows", None, "display.max_columns", None)
result_csv = pd.DataFrame(columns=[
    'id',
    'query',
    'Coverage Constraint',
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
    'qcut_time_sample',
    'qcut_mean_summed_time'
])

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)

    s = pd.Series(
        [int(filename[filename.find('Q'):].strip('.csv').strip('Q')), df['query'].iloc[0], df['CC'].iloc[0],
         df['card_true_tot_Q'].iloc[0], df['card_true_sa_Q'].iloc[0],
         df['card_true_tot_newQ'].iloc[0],
         df['card_true_sa_newQ'].iloc[0], df['proximity'].iloc[0], df['relaxation_degree'].iloc[0],
         df['disparity_Q'].iloc[0], df['fairness_Q'].iloc[0],
         df['time_preprocessing'].mean(), df['time_pruning'].mean(),
         df['time_algo'].mean(), df['time_sample'].mean(), df['time_tot'].mean()],
        index=result_csv.columns)
    result_csv = result_csv.append(s, ignore_index=True)

result_csv.sort_values(by="id", inplace=True)

result_csv.to_csv(r'C:\Users\Nicolò\Desktop\Tesi\res_tesi\df_risultati\rewriting_res.csv', index=False)
writer = pd.ExcelWriter(r'C:\Users\Nicolò\Desktop\Tesi\res_tesi\df_risultati\rewriting_res.xlsx')
result_csv.to_excel(writer)
writer.save()
