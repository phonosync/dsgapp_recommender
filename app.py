import streamlit as st
import io
import numpy as np
import pandas as pd
from surprise import BaselineOnly, Dataset, Reader, accuracy, SVD, KNNBasic
from surprise.dump import dump, load
from surprise.model_selection import train_test_split

import surprise_utils as s_utils

TRAINED = False

st.title("Recommender System")

st.write('''Train a recommender system with collaborative filtering approaches. Ratings for
         user-item pairs need to be provided for training:''')
df_styler = pd.DataFrame({'USER_ID': ['ab$956', 'kl73fv', 'xx98gh'], 'ITEM_ID': [0, 1, 1], 
                          'RATING': [4, 1, 3]}).style.hide()
st.table(data=df_styler)  
         
st.write('''Calculated recommendations will be clipped to the scale 
         [minimum rating, maximum rating]:''')
col1, col2 = st.columns(2)
with col1:
    rating_min = st.number_input(label='Minimum Rating', value=1)

with col2:
    rating_max = st.number_input(label='Maximum Rating', value=5)

inp_file = st.file_uploader(label='''Upload a csv-file (delimiter ';') with one row per rating and three columns in order 
         <USER_ID>, <ITEM_ID>, <RATING>. The first row is assumed to contain column headers 
         and no rating information:''')

data = None

if inp_file:

    df = pd.read_csv(inp_file, delimiter=';', header=0, names=['USER_ID', 'ITEM_ID', 'RATING'])
    
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(rating_min, rating_max))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["USER_ID", "ITEM_ID", "RATING"]], reader)

algo = None
model_type = st.selectbox('Select  model type:', ('kNN - cosine','SVD'))

if model_type == 'kNN - cosine':
    user_based = st.checkbox('''User based: Select if case similarities should be computed 
                                between users. Deselect to compute between items, 
                                i.e. item based-approach).''', value=True)
    
    k = st.number_input(label='k: Number of nearest neighbours to include', value=10)

    sim_options = {
        "name": "cosine",
        "user_based": user_based, 
    }
    algo = KNNBasic(k=k, min_k=1, sim_options=sim_options)

if model_type == 'SVD':
    algo=SVD()

if algo and data:
    
    # 1. train and evaluate 
    test_size = st.number_input(label='Size of test set', value=0.2)

    at_k = st.number_input(label='Precision/recall at k', value=5)
    at_k_threshold = st.number_input(label='Threshold for precision/recall at k', value=3.5)

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
    
    metrics['AvPrec@k'] = sum(prec for prec in precisions.values()) / len(precisions)
    metrics['AvRec@k'] = sum(rec for rec in recalls.values()) / len(recalls)

    st.table(metrics)

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

    

    st.number_input(label='Select number of recommendations to produce per user:', value=5)

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
        label='Download recommendations per user',
        data=buffer,
        file_name='recommendations.xlsx',
        mime='application/vnd.ms-excel',
    )   

    with open('algo.pickle', 'rb') as f:
        st.download_button('Download serialised model', f, file_name='algo.pickle')  # Defaults to 'application/octet-stream'
    # st.download_button(
    #     label="Download serialised model",
    #     data=buffer_str,
    #     file_name='algo.dump'#,
    #     # mime='application/vnd.ms-excel',
    # )

     
