from typing import List, Any, Dict, Union
import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import pandas.io.sql as sqlio
from datetime import timedelta
import psycopg2
import seaborn as sns
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
def transform_Q_to_compQ (query: str):
    """funzione che presa in input una query la trasforma per renderla la complementare"""
    if query.find('&') == -1 and (query.find('|') == -1):
        query = change_op_to_compQ(query)
    if "&" in query:
        query = query.replace("&", "|")
    else:
        query = query.replace("|", "&")
    query = change_op_to_compQ(query)

    return query


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
    """funzione che calcola per ogni elemento di CC la cvardinalità del gruppo protetto sul dataframe passato"""
    sens_values_count: List[int] = []
    for cond in k:
        condition = (' & ').join(
            [sens_attr + ' == \'' + cond['value'][x] + '\'' for x, sens_attr in enumerate(cond['AS'])])
        sens_values_count.append(sample.query(condition).shape[0])

    return sens_values_count


def get_comp_query_result (CC: object, indice: int, com_orig_query: str, df):
    """funzione che produce lo spazio in cui l'albero cercherà i k vicini per il risultato finale"""

    complementary_query_res = df.query(
        com_orig_query)  # complementare della query originale, quindi lo spazio in cui devo cercare i restanti elementi per arrivare al valore indicato nel CC

    condition_str = (' & ').join(
        [sens_attr + ' == \'' + CC[indice]['value'][x] + '\'' for x, sens_attr in enumerate(CC[indice]['AS'])])
    print(condition_str)
    complementary_query_AS = complementary_query_res.query(condition_str)
    sens_comp_q_value_count = counts_Value_col(complementary_query_AS, CC)
    num_values_AS = sens_comp_q_value_count[indice]

    space = complementary_query_AS.to_numpy()  # trasformo con numpy il dataframe per poterlo passare alla funzione KDtree# spazio su cui cercare i k vicini
    return num_values_AS, space


# trasformare le reloax_attr in liste così che sia più generalizzato
def attr_val_to_list (relxattr, list_columns):
    """funzione ausiliare per creare le liste utili all'algoritmo """
    list_attr = []  # lista degli attributi del predicato
    list_val_pred = []  # lista dei valori
    for i in relxattr:
        list_attr.append(i['attr'])
        list_val_pred.append(i['val'])

    test_col = []
    for i in list_attr:
        test_col.append(list_columns.index(i))

    return list_attr, test_col, list_val_pred


def knn_cov_cons (num_to_be_found, val_pred, space, norm_data):
    """
  #funzione che cerca gli attributi mancanti per soddisfare il vincolo di copertura
    :param num_to_be_found: numeri di vicini da trovare per raggiunger il CC
    :param space: lista degli elementi utili estratti dalla query originale
    :param val_pred: lista che rappresenta il punto da cui devo cercare i vicini
    :param norm_data: matrice con i dati normalizzati
    :return: numpy.ndarray dei k vicini

  """
    # normalizzazione kdtree deve lavorare sulle colonne normalizzate
    print("norm_val", val_pred)
    print(norm_data[:, 0].min())
    print(norm_data[:, 0].max())
    print(norm_data[:, 1].min())
    print(norm_data[:, 1].max())
    plt.scatter(norm_data[:, 0], norm_data[:, 1])
    plt.title('dati normalizzati')
    plt.show()
    tree = KDTree(norm_data)  # creo l'albero KDtree sullo spazio esterno in cui poi cercare gli elementi più vicini
    dist, ind = tree.query(val_pred, k=num_to_be_found)
    return space[ind[0]]


def norm_point (point_to_normalise, relax_attr):
    point_norm = np.zeros(len(point_to_normalise))
    for i in range(len(point_to_normalise)):
        point_norm[i] = np.divide(int(point_to_normalise[i]) - relax_attr[i]['val_min_col'],
                                  relax_attr[i]['val_max_col'] - relax_attr[i]['val_min_col'])
    return point_norm


def normalize_data (to_be_normalised):
    """funzione che normalizza le colonne in cui andare a cercare i k vicini 
    :param to_be_normalised: matrice dei dati da normalizzare
    :param relax_attributes: oggetto che mantiene le informazioni della query iniziale
    """
    # il problema secondo me è lo scambio tra massimo e minino, infatti il grafico viene al contrario nel caso i segni nella query siano > o >= mentre nell'altro caso il grafico si mantiene uguale(a quello senza normalizzazione) solo scalato come è giusto che sia
    for i in range(to_be_normalised.shape[1]):
        min = to_be_normalised[:, i].min()
        max = to_be_normalised[:, i].max()
        for j in range(to_be_normalised.shape[0]):
            to_be_normalised[j][i] = np.abs(np.divide(to_be_normalised[j][i] - min, max - min))
    return to_be_normalised


def get_new_val_for_pred (knn_found, attr_of_pred, relax_attributes):
    """funzione che trova il massimo o il minimo a seconda del caso per poi scrivere la query indotta  """
    list_res = []

    for indx in range(len(relax_attributes)):
        if relax_attributes[indx]['op'] == '<':
            list_res.append(knn_found[attr_of_pred[indx]].max() + 1)
        elif relax_attributes[indx]['op'] == '<=':
            list_res.append(knn_found[attr_of_pred[indx]].max())
        elif relax_attributes[indx]['op'] == '>':
            if knn_found[attr_of_pred[indx]].min() <= 0:
                list_res.append(knn_found[attr_of_pred[indx]].min())
                continue
            list_res.append(knn_found[attr_of_pred[indx]].min() - 1)
        elif relax_attributes[indx]['op'] == '>=':
            list_res.append(knn_found[attr_of_pred[indx]].min())
    return list_res


def induced_query (query, list_res, list_val_pred):
    """funzione che crea la query indotta"""
    zipped_list = list(zip(list_val_pred[0],
                           list_res))  # unisco la lista dei vecchi valori dei predicati della query iniziale con i valori massimi/minimi con cui costruire la query indotta
    for i in zipped_list:
        query = query.replace(i[0], str(i[1]))
    return query


def get_relaxation_degree (card_Q, card_new_Q):
    """ cardinalità totale iniziale e finale per il calcolo della misura relax_degree """
    return (card_new_Q - card_Q) / card_Q


def get_fairness_index (card_Q, card_new_Q, card_AS_Q, card_AS_Qnew):
    """ cardinalità totale iniziale e finale , cardinalità del gruppo protetto iniziale e finale """
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

    return np.linalg.norm(point_q - point_qind)


def max_min_per_column (df, relax_attr):
    max_list = []
    min_list = []
    for i in relax_attr:
        curr_max = df[i['attr']].max()
        curr_min = df[i['attr']].min()
        max_list.append(curr_max)
        min_list.append(curr_min)
    for i in range(len(relax_attr)):
        relax_attr[i]['val_max_col'] = max_list[i]
        relax_attr[i]['val_min_col'] = min_list[i]


def read_table (c, table):  # , attributes
    sql = 'SELECT * FROM ' + table
    # sql = 'SELECT ' + ', '.join(attributes) + ' FROM ' + table + ' ORDER BY RANDOM() LIMIT ' + str(sample_size)
    return sqlio.read_sql_query(sql, c)


def get_table (query):
    m = re.match(r'.*FROM\s(.*)\sWHERE(.*)', query)
    sens_attr_table = m.group(1)
    return sens_attr_table


def knn_res_Qind (query, CC, df, list_columns):
    start_time = time.time()
    list_temp = []
    orig_query, relax_attributes = query_pred_tokenize(query)
    max_min_per_column(df, relax_attributes)
    print(relax_attributes)
    initial_query_results = df.query(orig_query)
    card_AS_Q: List[int] = counts_Value_col(initial_query_results, CC)
    list_attr, list_col_pred, list_val_pred = attr_val_to_list(relax_attributes, list_columns)
    point_normalized = norm_point(list_val_pred, relax_attributes)
    point_normalized = [point_normalized]
    list_val_pred = [list_val_pred]
    card_tot_Q = initial_query_results.shape[0]
    print("card_tot_Q", card_tot_Q)
    print("card_AS_Q", card_AS_Q)
    com_orig_query = transform_Q_to_compQ(
        orig_query)
    print(com_orig_query)
    time_tree_exec = 0.0
    for indx in range(len(card_AS_Q)):
        if card_AS_Q[indx] >= int(CC[indx]['num']):
            if (len(CC) == 1):
                test_res_temp = [len(relax_attributes), query, card_tot_Q, card_AS_Q, CC, '', '', '', '', '', '', '',
                                 '', '', '', '', '', '', '', '', '']
                return test_res_temp
            continue
        card_AS_Qcomp, space = get_comp_query_result(CC, indx, com_orig_query, df)
        norm_data = normalize_data(space[:, list_col_pred])
        num_to_be_found = int(CC[indx]['num']) - card_AS_Q[indx]
        if num_to_be_found > card_AS_Qcomp:
            res_space = pd.DataFrame(space, columns=list_columns)
            list_temp.append(res_space)
            continue
        start_time_tree = time.time()
        knn_points = knn_cov_cons(num_to_be_found, point_normalized, space, norm_data)
        time_tree_exec = (time.time() - start_time_tree)
        res_space = pd.DataFrame(knn_points, columns=list_columns)
        print(res_space)
        list_temp.append(
            res_space)
    #### RES APPROCCIO ESECUZIONE ####
    res_space_no_duplicate = pd.concat(list_temp).drop_duplicates().reset_index(drop=True)
    # lista con tutti i k vicini trovati per ogni CC da unire evitando di avere doppioni
    last_result = initial_query_results.append(res_space_no_duplicate)

    card_AS_res: List[int] = counts_Value_col(last_result, CC)
    card_tot_res = last_result.shape[0]
    print("Card_tot_res", card_tot_res)
    print("card_as_res", card_AS_res)
    time_exec = time.time() - start_time

    #### QUERY INDOTTA ####
    start_time_Qind = time.time()
    list_res = get_new_val_for_pred(res_space_no_duplicate, list_attr,
                                    relax_attributes)

    induced_Q_inSQL = induced_query(query, list_res, list_val_pred)
    print(induced_Q_inSQL)
    time_exec_Qind = time.time() - start_time_Qind
    induced_Q, relax_attributes_Qind = query_pred_tokenize(induced_Q_inSQL)
    df_induced_query = df.query(induced_Q)
    card_tot_Qind = df_induced_query.shape[0]
    print("Card_Tot_Qind", card_tot_Qind)
    card_AS_Qind: List[int] = counts_Value_col(df_induced_query, CC)
    print("Card_AS_Qind", card_AS_Qind)
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
                     measure_DispInd(card_AS_Q, card_tot_res),
                     get_relaxation_degree(card_tot_Q, card_tot_Qind),
                     measure_FairInd(card_tot_Q, card_tot_Qind, card_AS_Q, card_AS_Qind),
                     measure_DispInd(card_AS_Q, card_tot_Qind),
                     proximity(relax_attributes, relax_attributes_Qind)
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
                                            'time_read_table'
                                            ])
    # CC = [{'AS': ['sex'], 'value': ['Female'], 'num': '13689'}]
    CC = [{'AS': ['race'], 'value': ['Black'], 'num': '60'}]
    # for i in range(len(CC)):
    # CC[i]['num'] = str(input("inserisci il numero da raggiungere per il constraint: "))
    df = pd.read_csv('C:/Users/Nicolò/Desktop/Tesi/adult_data.csv')
    # fig, scatter = plt.subplots(figsize=(10, 6), dpi=100)
    # scatter =sns.scatterplot(data=df , x="hours_per_week", y ="education_num" , hue="race")
    # plt.show()
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    np.set_printoptions(threshold=np.inf)
    list_columns = list(
        df.columns)
    query = 'SELECT * FROM adult_data WHERE education_num >= 14 AND hours_per_week >40'  # str(input("Inserire la query: "))' SELECT * FROM adult_data WHERE capital_loss < 1500 AND age < 55'

    # c = psycopg2.connect(host='localhost', port=5432, user='Insert_Here_Your_User',
    # password='Insert_Here_Your_Password', database='postgres')
    time_start_read_table = time.time()
    # df = read_table(c , get_table(query) )
    time_end_read_table = (time.time() - time_start_read_table)
    test_res_temp = knn_res_Qind(query, CC, df, list_columns)
    test_res_temp += [time_end_read_table]
    s = pd.Series(test_res_temp, index=test_result_csv.columns)
    test_result_csv = test_result_csv.append(s, ignore_index=True)
    # test_result_csv.to_csv('file1.csv', index=False)


if __name__ == '__main__':
    main()
