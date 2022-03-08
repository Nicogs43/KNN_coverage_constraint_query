import pandas as pd
import glob
path = r'C:\Users\Nicolò\Desktop\Tesi\res - execution'  # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
pd.set_option("display.max_rows", None, "display.max_columns", None)

frame.to_csv(r'C:\Users\Nicolò\Desktop\Tesi\res - execution\All_res_execution_dataframes.csv', index=False)
