import streamlit as st
import io
import pandas as pd
import csv
from surprise import Dataset, Reader, accuracy, SVD, KNNBasic
from surprise.dump import dump
from surprise.model_selection import train_test_split

import surprise_utils as s_utils

TRAINED = False

st.title("Recommender System")

st.write('''App zur Erstellung eines Recommender-Systems basierend auf Kollaborativen Filtern-Verfahren.
         Ratings für User-Item Paare müssen als Trainingsdatensatz zur Verfügung gestellt werden:''')
df_styler = pd.DataFrame({'USER_ID': ['ab$956', 'kl73fv', 'xx98gh'], 'ITEM_ID': [0, 1, 1], 
                          'RATING': [4, 1, 3]}).style.hide()
st.table(data=df_styler)  
         
st.write('''Berechnete Empfehlungen werden auf die Skala [minimum rating, maximum rating] geclippt:''')
col1, col2 = st.columns(2)
with col1:
    rating_min = st.number_input(label='Minimum Rating', value=1)

with col2:
    rating_max = st.number_input(label='Maximum Rating', value=5)

inp_file = st.file_uploader(label='''Laden Sie eine csv-Datei hoch bestehend aus einer Zeile pro
                            Rating und drei Spalten in der Reihenfolge <USER_ID>, <ITEM_ID>, <RATING>.
                            Die erste Zeile wird als Spaltenbezeichnungen interpretiert und kein
                            Rating wird ausgelesen:''')

data = None

if inp_file:

    # We always assume column labels (see format description above)
    header = 0
    bla = io.StringIO(inp_file.getvalue().decode("utf-8"))
    dialect = csv.Sniffer().sniff(bla.read())
    df = pd.read_csv(inp_file, dialect=dialect, header=header, names=['USER_ID', 'ITEM_ID', 'RATING'])

    # df = pd.read_csv(inp_file, delimiter=';', header=0, names=['USER_ID', 'ITEM_ID', 'RATING'])
    
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(rating_min, rating_max))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["USER_ID", "ITEM_ID", "RATING"]], reader)

algo = None
model_type = st.selectbox('Wählen Sie den Algorithmus:', ('kNN - cosine','SVD'))

if model_type == 'kNN - cosine':
    user_based = st.checkbox('''User-basiert: Setzen Sie einen Haken, falls die Ähnlichkeiten
                             zwischen Usern berechnet werden sollen. Heben Sie die Auswahl auf
                             für den Item-basierten Ansatz.''', value=True)
    
    k = st.number_input(label='k: Anzahl zu berücksichtigender nächster Nachbarn', value=10)

    sim_options = {
        "name": "cosine",
        "user_based": user_based, 
    }
    algo = KNNBasic(k=k, min_k=1, sim_options=sim_options)

if model_type == 'SVD':
    algo=SVD()

if algo and data:
    
    # 1. train and evaluate 
    test_size = st.number_input(label='Anteilsmässige Grösse des Testdatensatzes', value=0.2)

    at_k = st.number_input(label='Wählen Sie *k* für Precision/Recall @*k*', value=5)
    at_k_threshold = st.number_input(label='Threshold für die Berechnung von Precision/Recall @*k*', value=3.5)

    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
    
    algo.fit(trainset)
    predictions_testset = algo.test(testset)
    # Then compute accuracy metrics
    metrics = {
        'RMSD': accuracy.rmse(predictions_testset),
        'MAE': accuracy.mae(predictions_testset),
        'MSE': accuracy.mse(predictions_testset)
    }

    precisions, recalls = s_utils.precision_recall_at_k(predictions_testset, k=at_k, 
                                                        threshold=at_k_threshold)
    
    metrics['MAP@k'] = sum(prec for prec in precisions.values()) / len(precisions)
    metrics['MAR@k'] = sum(rec for rec in recalls.values()) / len(recalls)

    metrics_to_print = {
        'Metric': [],
        'Value': []
    }
    for k,v in metrics.items():
        metrics_to_print['Metric'].append(k)
        metrics_to_print['Value'].append('{:,.2f}'.format(v))

    st.table(pd.DataFrame(metrics_to_print).style.hide())

    metrics['k'] = at_k
    metrics['@k threshold'] = at_k_threshold
    metrics['Test size'] = test_size

    # 2. train on full dataset
    trainset_full = data.build_full_trainset()
    algo.fit(trainset_full)

    # buffer_algo = io.StringIO()
    # buffer_str = 
    dump('algo.pickle', predictions=None, algo=algo, verbose=0)

    # Then predict ratings for all pairs (u, i) that are NOT in the training set.
    predictions = algo.test(trainset_full.build_anti_testset())

    

    st.number_input(label='Wählen Sie die Anzahl zu berechnender Empfehlungen pro User:', value=5)

    top_n = s_utils.get_top_n(predictions, n=5)

    recommendations = []
    for u_id, item_ratings in top_n.items():
        for i_id, rating in item_ratings:
            recommendations.append([u_id, i_id, rating])
    df_recs = pd.DataFrame(recommendations, columns=['USER_ID', 'ITEM_ID', 'RATING'])

    # pd.DataFrame({k: [v] for k,v in metrics.items()})

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:  
        df_recs.to_excel(writer, sheet_name='Recommendations', index=False)
        pd.DataFrame({k: [v] for k,v in metrics.items()}).to_excel(writer, sheet_name='Accuracy metrics', index=False)


    st.download_button(
        label='Download der berechneten Empfehlungen pro User',
        data=buffer,
        file_name='recommendations.xlsx',
        mime='application/vnd.ms-excel',
    )   

    with open('algo.pickle', 'rb') as f:
        st.download_button('Download des serialisierten Modells', f, file_name='algo.pickle')  # Defaults to 'application/octet-stream'
    # st.download_button(
    #     label="Download serialised model",
    #     data=buffer_str,
    #     file_name='algo.dump'#,
    #     # mime='application/vnd.ms-excel',
    # )

     
