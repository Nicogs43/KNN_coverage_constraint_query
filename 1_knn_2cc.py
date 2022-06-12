# -*- coding: utf-8 -*-
import math
import re
import time
from datetime import timedelta
from typing import List, Any, Dict, Union

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pandas.io.sql as sqlio
import psycopg2
from sklearn.neighbors import KDTree
# import seaborn as sns
from sklearn.preprocessing import normalize


def query_pred_tokenize (query: str):
    # we extract the information we need from the query and we reorganize them
    # print('Query originale:\n\n' + query + '\n')
    m = re.match(r'.*FROM\s(.*)\sWHERE(.*)', query)
    # sens_attr_table = m.group(1)

    relax_attributes: List[Dict[str, str]] = []
    boolean_op = ''

    if (' OR ' in m.group(2) or ' or ' in m.group(2)) and (' AND ' in m.group(2) or ' and ' in m.group(2)):
        print('Conditions must be all ANDs or all ORs!')
        exit(1)
    elif ' OR ' in m.group(2):
        boolean_op = ' | '
    elif ' AND ' in m.group(2):
        boolean_op = ' & '

    for cond in re.split(' AND | OR ', m.group(2)):
        c = re.match(r'([a-zA-Z\d\_]+)\s*(\<|\>|\<\=|\>\=)\s*([\-]?[\d]+[\.]?[\d]*)', cond.strip())
        relax_attributes.append(
            {
                'attr': c.group(1).strip(),
                'op': c.group(2).strip(),
                'val_orig': c.group(3).strip(),
                'val': c.group(3).strip()
            })

    orig_query: str = boolean_op.join([x['attr'] + ' ' + x['op'] + ' ' + x['val_orig'] for x in relax_attributes])

    return orig_query, relax_attributes


# funzione che crea la complementare della orig_query
def transform_Q_to_compQ (query: str, CC: object, indice: int):
    # """funzione che presa in input una query la trasforma per renderla la complementare"""
    if query.find('&') == -1 and (query.find('|') == -1):
        query = change_op_to_compQ(query)
    if "&" in query:
        query = query.replace("&", "|")
    else:
        query = query.replace("|", "&")
    query = change_op_to_compQ(query)

    condition_str = (' & ').join(
        [sens_attr + ' == \'' + CC[indice]['value'][x] + '\'' for x, sens_attr in enumerate(CC[indice]['AS'])])

    comp_q = '(' + query + ') & (' + condition_str + ')'

    return comp_q


def change_op_to_compQ (query: str):
    list_str = query.split()
    for i in range(len(list_str)):
        if list_str[i] == '<':
            list_str[i] = '>='
        elif list_str[i] == '>':
            list_str[i] = '<='
        elif list_str[i] == '>=':
            list_str[i] = '<'
        elif list_str[i] == '<=':
            list_str[i] = '>'
    query = " ".join(list_str)
    return query


def counts_Value_col (sample, k: object):
    # """funzione che calcola per ogni elemento di CC la cvardinalità del gruppo protetto sul dataframe passato"""
    sens_values_count: List[int] = []
    for cond in k:
        condition = (' & ').join(
            [sens_attr + ' == \'' + cond['value'][x] + '\'' for x, sens_attr in enumerate(cond['AS'])])
        sens_values_count.append(sample.query(condition).shape[0])

    return sens_values_count


def get_comp_query_result (CC: object, indice: int, comp_q: str, df):
    # """funzione che produce lo spazio in cui l'albero cercherà i k vicini per il risultato finale"""

    complementary_query_AS = df.query(comp_q)
    sens_comp_q_value_count = counts_Value_col(complementary_query_AS, CC)
    num_values_AS = sens_comp_q_value_count[indice]

    space = complementary_query_AS.to_numpy()  # trasformo con numpy il dataframe per poterlo passare alla funzione KDtree# spazio su cui cercare i k vicini
    return num_values_AS, space


# trasformare le reloax_attr in liste così che sia più generalizzato
def attr_val_to_list (relxattr, list_columns):
    # """funzione ausiliare per creare le liste utili all'algoritmo """
    list_attr = []  # lista degli attributi del predicao
    list_val_pred = []  # lista dei valori
    for i in relxattr:
        list_attr.append(i['attr'])
        list_val_pred.append(i['val'])

    test_col = [list_columns.index(i) for i in list_attr]

    return list_attr, test_col, list_val_pred


def knn_cov_cons (num_to_be_found, val_pred, space, norm_data, list_columns):
    #   """
    # #funzione che cerca gli attributi mancanti per soddisfare il vincolo di copertura
    #   :param num_to_be_found: numeri di vicini da trovare per raggiunger il CC
    #   :param space: lista degli elementi utili estratti dalla query originale
    #   :param val_pred: lista che rappresenta il punto da cui devo cercare i vicini
    #   :param norm_data: matrice con i dati normalizzati
    #   :return: numpy.ndarray dei k vicini
    #
    # """

    tree = KDTree(norm_data)  # creo l'albero KDtree sullo spazio esterno in cui poi cercare gli elementi più vicini
    dist, ind = tree.query(val_pred, k=num_to_be_found)

    res_space = pd.DataFrame(space[ind[0]], columns=list_columns)
    return res_space


def norm_point (point_to_normalise, relax_attr):
    point_norm = np.zeros(len(point_to_normalise))
    for i in range(len(point_to_normalise)):
        point_norm[i] = np.divide(int(point_to_normalise[i]) - relax_attr[i]['val_min_col'],
                                  relax_attr[i]['val_max_col'] - relax_attr[i]['val_min_col'])
    return [point_norm]


def normalize_data (to_be_normalised, relax_attributes):
    # """funzione che normalizza le colonne in cui andare a cercare i k vicini
    # :param to_be_normalised: matrice dei dati da normalizzare
    # :param relax_attributes: oggetto che mantiene le informazioni della query iniziale
    # """

    min = np.array([elem['val_min_col'] for elem in relax_attributes])
    max = np.array([elem['val_max_col'] for elem in relax_attributes])

    aa = (to_be_normalised - min) / (max - min)
    return aa


def get_new_val_for_pred (knn_found, attr_of_pred, relax_attributes):
    # """funzione che trova il massimo o il minimo a seconda del caso per poi scrivere la query indotta  """
    list_res = []
    for indx in range(len(relax_attributes)):
        if relax_attributes[indx]['op'] == '<':
            list_res.append(max(int(relax_attributes[indx]['val']), knn_found[attr_of_pred[indx]].max() + 1))
        elif relax_attributes[indx]['op'] == '<=':
            list_res.append(max(int(relax_attributes[indx]['val']), knn_found[attr_of_pred[indx]].max()))
        elif relax_attributes[indx]['op'] == '>':
            if knn_found[attr_of_pred[indx]].min() <= 0:
                list_res.append(min(int(relax_attributes[indx]['val']), knn_found[attr_of_pred[indx]].min()))
                continue
            list_res.append(min(int(relax_attributes[indx]['val']), knn_found[attr_of_pred[indx]].min() - 1))
        elif relax_attributes[indx]['op'] == '>=':
            list_res.append(min(int(relax_attributes[indx]['val']), knn_found[attr_of_pred[indx]].min()))
    return list_res


def induced_query (query, list_res, list_val_pred):
    # """funzione che crea la query indotta"""
    zipped_list = list(zip(list_val_pred[0],
                           list_res))  # unisco la lista dei vecchi valori dei predicati della query iniziale con i valori massimi/minimi con cui costruire la query indotta
    for i in zipped_list:
        query = query.replace(i[0], str(i[1]))
    return query


def get_relaxation_degree (card_Q, card_new_Q):
    # """ cardinalità totale iniziale e finale per il calcolo della misura relax_degree """
    return (card_new_Q - card_Q) / card_Q


def get_fairness_index (card_Q, card_new_Q, card_AS_Q, card_AS_Qnew):
    # """ cardinalità totale iniziale e finale , cardinalità del gruppo protetto iniziale e finale """
    initial_fairness = card_AS_Q / (card_Q - card_AS_Q)
    new_fairness = card_AS_Qnew / (card_new_Q - card_AS_Qnew)
    return new_fairness - initial_fairness


def get_disparity_index (card_AS_Q, card_new_Q):
    return card_AS_Q / card_new_Q


def measure_DispInd (card_AS_Q, card_tot_Q):
    res = []
    for i in card_AS_Q:
        res.append(get_disparity_index(i, card_tot_Q))
    return res


def measure_FairInd (card_tot_Q, card_tot_newQ, card_AS_Q, card_AS_Qnew):
    res = []
    for i, j in zip(card_AS_Q, card_AS_Qnew):
        res.append(get_fairness_index(card_tot_Q, card_tot_newQ, i, j))
    return res


def proximity (relax_attr, relax_attr_Qind):
    point_q = np.zeros(len(relax_attr))
    point_qind = np.zeros(len(relax_attr))
    for i in range(len(relax_attr)):
        if (relax_attr[i]['op'] == '<') | (relax_attr[i]['op'] == '<='):
            point_q[i] = np.divide(int(relax_attr[i]['val']) - int(relax_attr[i]['val']),
                                   relax_attr[i]['val_max_col'] - int(relax_attr[i]['val']))
            point_qind[i] = np.divide(int(relax_attr_Qind[i]['val']) - int(relax_attr[i]['val']),
                                      relax_attr[i]['val_max_col'] - int(relax_attr[i]['val']))
        elif (relax_attr[i]['op'] == '>') | (relax_attr[i]['op'] == '>='):
            point_q[i] = np.divide(int(relax_attr[i]['val']) - int(relax_attr[i]['val']),
                                   relax_attr[i]['val_min_col'] - int(relax_attr[i]['val']))
            point_qind[i] = np.divide(int(relax_attr_Qind[i]['val']) - int(relax_attr[i]['val']),
                                      relax_attr[i]['val_min_col'] - int(relax_attr[i]['val']))
    if np.all(point_q != 0):
        print("Il punto della query iniziale non si trova nell'origine. Qualcosa deve essere andato storto!")

    dist = 0.0
    for ii, ee in enumerate(point_q):
        dist += (point_qind[ii]) ** 2
    dist = math.sqrt(dist)
    # return np.linalg.norm(point_q - point_qind)
    return dist / math.sqrt(len(relax_attr))


def max_min_per_column (df, relax_attr):
    for i in range(len(relax_attr)):
        relax_attr[i]['val_max_col'] = df[relax_attr[i]['attr']].max()
        relax_attr[i]['val_min_col'] = df[relax_attr[i]['attr']].min()


def read_table (c, table):  # , attributes
    sql = 'SELECT * FROM ' + table
    # sql = 'SELECT ' + ', '.join(attributes) + ' FROM ' + table + ' ORDER BY RANDOM() LIMIT ' + str(sample_size)
    return sqlio.read_sql_query(sql, c)


def get_table (query):
    m = re.match(r'.*FROM\s(.*)\sWHERE(.*)', query)
    sens_attr_table = m.group(1)
    return sens_attr_table


def min_max (sample, attrs):
    result = []
    for a in attrs:
        result += [{'attr': a, 'min': sample[a].min(), 'max': sample[a].max()}]
    return result


def to_code (string):
    code = "(" + string + ")"
    code = re.sub("[a-z_]+", lambda m: "df['%s']" % m.group(0), code)
    code = code.replace("&", ") & (")
    code = code.replace("|", " ) | ( ")
    # code = code.replace("=", "==")
    return code


def knn_res_Qind (query, CC, df, list_columns):
    time_tree_exec = 0.0
    list_temp = []
    start_time = time.time()
    #################### useful data manipulation
    m = re.match(r'.*FROM\s(.*)\sWHERE(.*)', query)

    relax_attributes: List[Dict[str, str]] = []
    boolean_op = ''
    if ' AND ' in m.group(2):
        boolean_op = ' & '
    else:
        boolean_op = ' | '

    for cond in re.split(' AND | OR ', m.group(2)):
        c = re.match(r'([a-zA-Z\d\_]+)\s*(\<|\>|\<\=|\>\=)\s*([\-]?[\d]+[\.]?[\d]*)', cond.strip())
        relax_attributes.append(
            {
                'attr': c.group(1).strip(),
                'op': c.group(2).strip(),
                'val_orig': c.group(3).strip(),
                'val': c.group(3).strip()
            })

    orig_query: str = boolean_op.join([x['attr'] + ' ' + x['op'] + ' ' + x['val_orig'] for x in relax_attributes])
    cond_cc = [(' & ').join(["(complementary['" + sens_attr + "'] == '" + cond['value'][x] + '\')' for x, sens_attr in
                             enumerate(cond['AS'])]) for cond in CC]
    cond_cc2 = [(' & ').join(
        ["(initial_query_results['" + sens_attr + "'] == '" + cond['value'][x] + '\')' for x, sens_attr in
         enumerate(cond['AS'])]) for cond in CC]

    max_min_per_column(df, relax_attributes)
    list_attr, list_col_pred, list_val_pred = attr_val_to_list(relax_attributes, list_columns)
    time1 = time.time()
    #### we find the result set and the complementary set #### qui creo df e complementary che mi servono
    complementary, initial_query_results = [xx for _, xx in
                                            df.groupby(eval(to_code(orig_query)))]  # else xx[xx['sex']=='Female']
    bb2 = time.time()
    complementary_spaces = [complementary.loc[eval(cond_cc[i])] for i in range(0, len(cond_cc))]
    time1_2 = time.time()

    ## we count how many individuals for each protected group are in the result
    card_AS_Q = [len(initial_query_results.loc[eval(cond_cc2[i])]) for i in range(0, len(cond_cc2))]
    time1_3 = time.time()

    point_normalized = norm_point(list_val_pred, relax_attributes)

    list_val_pred = [list_val_pred]
    card_tot_Q = initial_query_results.shape[0]

    time_norm_data = []
    time_tree_exec = []
    time_res_no_duplicates = []
    res_space_no_duplicate = pd.DataFrame()

    start_time_for = time.time()
    for indx in range(len(card_AS_Q)):
        if card_AS_Q[indx] >= int(CC[indx]['num']):
            if (len(CC) == 1):
                test_res_temp = [len(relax_attributes), query, card_tot_Q, card_AS_Q, CC, '', '', '', '', '', '', '',
                                 '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
                return test_res_temp
            continue

        # trasformo dati in matrice per knn e normalizzo
        start_time4 = time.time()
        card_AS_Qcomp = len(complementary_spaces[indx])
        space = complementary_spaces[
            indx].to_numpy()  # trasformo con numpy il dataframe per poterlo passare alla funzione KDtree# spazio su cui cercare i k vicini
        norm_data = normalize_data(space[:, list_col_pred], relax_attributes)  # 0.0089755
        start_time5 = time.time()
        time_norm_data.append(start_time5 - start_time4)

        num_to_be_found = int(CC[indx]['num']) - card_AS_Q[indx]
        if num_to_be_found > card_AS_Qcomp:
            res_space = pd.DataFrame(space, columns=list_columns)
            list_temp.append(res_space)
            continue
        start_time_tree = time.time()
        res_space = knn_cov_cons(num_to_be_found, point_normalized, space, norm_data, list_columns)
        end_time_tree = time.time()
        res_space_no_duplicate = pd.concat([res_space_no_duplicate, res_space]).drop_duplicates().reset_index(drop=True)
        time_6_1 = time.time()
        time_tree_exec.append(end_time_tree - start_time_tree)
        time_res_no_duplicates.append(time_6_1 - end_time_tree)

    end_time_for = time.time()
    #### RES APPROCCIO ESECUZIONE ####
    initial_query_results = pd.concat([initial_query_results, res_space_no_duplicate])
    end_time = time.time()

    time_for_df = bb2 - time1
    time_for_df_cc = time1_2 - bb2
    time_init_numero_cond = (time1 - start_time)
    time_query_as = time1_3 - time1_2

    time_finalres = (end_time - end_time_for)
    time_exec = end_time - start_time

    card_AS_res: List[int] = counts_Value_col(initial_query_results, CC)
    card_tot_res = initial_query_results.shape[0]

    if res_space_no_duplicate.empty:
        induced_Q_inSQL = query
    else:
        #### QUERY INDOTTA ####
        start_time_Qind = time.time()
        list_res: List[Union[int, Any]] = get_new_val_for_pred(res_space_no_duplicate, list_attr, relax_attributes)
        induced_Q_inSQL = induced_query(query, list_res, list_val_pred)

        time_exec_Qind = time.time() - start_time_Qind

    induced_Q, relax_attributes_Qind = query_pred_tokenize(induced_Q_inSQL)

    df_induced_query = df.query(induced_Q)
    card_tot_Qind = df_induced_query.shape[0]
    card_AS_Qind: List[int] = counts_Value_col(df_induced_query, CC)

    time_selection1 = time.time()
    initial_query_results = df.query(orig_query)
    time_selection2 = time.time()
    time_selection = time_selection2 - time_selection1

    test_res_temp = [len(relax_attributes),
                     query,
                     card_tot_Q,
                     card_AS_Q,
                     CC,
                     induced_Q_inSQL,
                     card_tot_res,
                     card_AS_res,
                     card_tot_res - card_tot_Q,
                     card_tot_Qind,
                     card_AS_Qind,
                     time_exec,
                     time_exec_Qind,
                     time_tree_exec,
                     get_relaxation_degree(card_tot_Q, card_tot_res),
                     measure_FairInd(card_tot_Q, card_tot_res, card_AS_Q, card_AS_res),
                     # measure_DispInd(card_AS_Q, card_tot_res),
                     measure_DispInd(card_AS_res, card_tot_res),
                     get_relaxation_degree(card_tot_Q, card_tot_Qind),
                     measure_FairInd(card_tot_Q, card_tot_Qind, card_AS_Q, card_AS_Qind),
                     # measure_DispInd(card_AS_Q, card_tot_Qind),
                     measure_DispInd(card_AS_Qind, card_tot_Qind),
                     proximity(relax_attributes, relax_attributes_Qind),
                     time_init_numero_cond,
                     time_for_df,
                     time_for_df_cc,
                     time_query_as,
                     # time_norm_point,
                     time_norm_data,
                     time_res_no_duplicates,
                     # time_for_CC,
                     # time_if,
                     time_finalres
                     ]

    return test_res_temp


def main ():
    test_result_csv = pd.DataFrame(columns=['n_condizioni',
                                            'query',
                                            'card_true_tot_Q',
                                            'card_true_sa_Q',
                                            'CC',
                                            'Q_ind',
                                            'card_tot_res',
                                            'card_AS_res',
                                            'card_CC_added',
                                            'card_true_tot_Qind',
                                            'card_true_sa_Qind',
                                            'time_execution',
                                            'time_exec_Qind',
                                            'time_tree_exec',
                                            'relaxation_degree_res',
                                            'fairness_index_res',
                                            'disparity_index_res',
                                            'relaxation_degree_Qind',
                                            'fairness_index_Qind',
                                            'disparity_index_Qind',
                                            'proximity_Qind',
                                            'time_init_numero_cond',
                                            'time_for_computing_df',
                                            'time_for_computing_df_cc',
                                            'time_query_as',
                                            # 'time_norm_point',
                                            'time_norm_data',
                                            # 'time_if',
                                            'time_res_no_duplicates',
                                            # 'time_for_CC',
                                            'time_finalres',
                                            'time_read_table',
                                            'time_list_col'
                                            ])

    # CC = [{'AS': ['sex', 'race'], 'value': ['Female', 'Black'], 'num': '680'}]
    # CC = [{'AS': ['race'], 'value': ['Black'], 'num': '1450'}]
    # CC = [{'AS': ['sex'], 'value': ['Female'], 'num': '780'}]
    # CC = [{'AS': ['sex'], 'value': ['Female'], 'num': '1450'}]
    # CC = [{'AS': ['sex','marital_status'], 'value': ['Female','Married-civ-spouse'], 'num': '280'}]
    # CC = [{'AS': ['marital_status'], 'value': ['Married-civ-spouse'], 'num': '5700'}]
    # CC = [{'AS': ['race'], 'value': ['Black'], 'num': '210'},{'AS': ['race'], 'value': ['Asian-Pac-Islander'], 'num': '210'}]
    # CC = [{'AS': ['race'], 'value': ['Black'], 'num': '210'},{'AS': ['race'], 'value': ['Asian-Pac-Islander'], 'num': '210'},{'AS': ['race'], 'value': ['Amer-Indian-Eskimo'], 'num': '20'}]
    # CC = [{'AS': ['sex','race'], 'value': ['Female','Black'], 'num': '680'}]
    # CC = [{'AS': ['sex'], 'value': ['Female'], 'num': '4400'}, {'AS': ['race'], 'value': ['Black'], 'num': '1450'}]
    # CC = [{'AS': ['race'], 'value': ['Asian-Pac-Islander'], 'num': '500'}]
    # CC = [{'AS': ['sex'], 'value': ['Female'], 'num': '9100'}]

    # CC = [{'AS': ['race'], 'value': ['Black'], 'num': '20'}]
    # CC = [{'AS': ['sex'], 'value': ['Female'], 'num': '200'}]
    # CC = [{'AS': ['sex'], 'value': ['Female'], 'num': '500'}]
    # CC = [{'AS': ['race'], 'value': ['Black'], 'num': '80'}]
    # CC = [{'AS': ['sex'], 'value': ['Female'], 'num': '6000'}]
    # CC = [{'AS' : ['gender'], 'value':['Female'], 'num': '10000'}] #5000

    # for i in range(len(CC)):
    #     CC[i]['num'] = str(input("inserisci il numero da raggiungere per il constraint: "))
    # df = pd.read_csv(r'C:\Users\Nicolò\Desktop\Tesi\adult_data.csv')
    # query = str(input("Inserire la query: "))

    # query = 'SELECT * FROM adult_data WHERE age <= 46 AND education_num >= 14'
    # query = 'SELECT * FROM adult_data WHERE hours_per_week > 40 AND age >= 38'
    # query = 'SELECT * FROM adult_data WHERE education_num < 12 AND hours_per_week <= 40'
    # query = 'SELECT * FROM adult_data WHERE education_num >= 13 AND age < 35 AND hours_per_week <= 40'
    # query = 'SELECT * FROM adult_data WHERE education_num >= 13 AND age < 55 AND capital_gain <= 3000'
    # query = 'SELECT * FROM adult_data WHERE age < 40 AND hours_per_week <= 40 AND capital_loss < 2500'
    # query = 'SELECT * FROM adult_data WHERE age < 35 AND education_num >= 13 AND hours_per_week <= 40 AND capital_loss < 1500'
    # query = 'SELECT * FROM adult_data WHERE capital_gain <= 1500 AND age <= 34 AND capital_loss <= 500 AND hours_per_week > 37'
    # query = 'SELECT * FROM adult_data WHERE education_num <= 13 AND hours_per_week >= 32 AND capital_gain < 500 AND age < 50'

    # query ='SELECT * FROM diabetic_data WHERE num_medications < 15 AND time_in_hospital > 5'
    # query = 'SELECT * FROM diabetic_data WHERE num_medications < 10 AND time_in_hospital < 5 AND number_emergency < 5'

    c = psycopg2.connect(host='localhost', port=5432, user='postgres', password='postgreSQL',
                         database='postgres')  # inserire i dati del proprio Db

    time_start_read_table = time.time()
    df = read_table(c, get_table(query))
    time_med = time.time()
    time_end_read_table = (time_med - time_start_read_table)
    list_columns = list(df.columns)
    time_list_col = (time.time() - time_med)
    for n_run in range(1):  # 10
        test_res_temp = knn_res_Qind(query, CC, df, list_columns)
        test_res_temp += [time_end_read_table, time_list_col]
        s = pd.Series(test_res_temp, index=test_result_csv.columns)
        test_result_csv = test_result_csv.append(s, ignore_index=True)
    test_result_csv.to_csv('file1.csv', index=False)
    test_result_csv.to_csv('res_maggio/execution_Q5_norm2.csv', index=False)

    writer = pd.ExcelWriter('res_maggio/execution_Q5_norm2.xlsx')
    test_result_csv.to_excel(writer)
    writer.save()


if __name__ == '__main__':
    main()
