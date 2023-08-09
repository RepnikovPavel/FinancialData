from scipy.stats import ks_2samp
import numpy as np
import pandas as pd

def get_min_max(table,column_name):
    return table[column_name].min(),table[column_name].max()
def group_by_time_interval(table:pd.DataFrame, column_name, time_intervals):
    groups = {}
    for i in range(len(time_intervals)):
        print('\r{}/{}'.format(i,len(time_intervals)-1),end='')
        t1,t2 = time_intervals[i]
        greather_then = table.loc[table[column_name] >= t1]
        if i == len(time_intervals)-1:
            greather_and_less = greather_then.loc[greather_then[column_name] <= t2]
            groups.update({time_intervals[i]: greather_and_less})
        else:
            greather_and_less = greather_then.loc[greather_then[column_name] < t2]
            groups.update({time_intervals[i]: greather_and_less})
    print('')
    return groups
def is_data_identical(table1,table2)->pd.DataFrame:
    p_min = 0.05
    d_ = {}
    siml_count = 0
    diff_count = 0
    for cName in table1:
        # print(cName)
        d1 = table1[cName].dropna()
        d2 = table2[cName].dropna()
        if len(d1)==0 or len(d2) ==0:
            d_.update({cName:['diff']})
            diff_count+=1
            continue
        # print(d1.info())
        # print(d2.info())
        stat_result = ks_2samp(d1, d2)

        if stat_result.pvalue > p_min:
            # distribution are same
            d_.update({cName:['siml']})
            siml_count+=1
        else:
            # distribution are different
            d_.update({cName:['diff']})
            diff_count+=1
    d_.update({'sim_count':[siml_count]})
    d_.update({'diff_count':[diff_count]})
    df = pd.DataFrame(data = d_)
    all_names = [el for el in df]
    first_cols = ['sim_count','diff_count']
    other = np.setdiff1d(all_names,first_cols).tolist()
    out_columns = first_cols+other
    return df[out_columns]
