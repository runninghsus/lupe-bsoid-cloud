import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def csv_predict(condition):
    predict_dict = {key: [] for key in range(len(st.session_state['features'][condition]))}
    predict_df = []
    for f in range(len(st.session_state['features'][condition])):
        predict = st.session_state['classifier'].predict(st.session_state['features'][condition][f])
        predict_dict[f] = {'condition': np.repeat(condition, len(predict)),
                           'file': np.repeat(f, len(predict)),
                           'time': np.round(np.arange(0, len(predict) * 0.1, 0.1), 2),
                           'behavior': predict}
        predict_df.append(pd.DataFrame(predict_dict[f]))
    concat_df = pd.concat([predict_df[f] for f in range(len(predict_df))])
    return convert_df(concat_df)


def duration_pie_csv(condition):
    predict_dict = {key: [] for key in range(len(st.session_state['features'][condition]))}
    duration_pie_df = []
    for f in range(len(st.session_state['features'][condition])):
        predict = st.session_state['classifier'].predict(st.session_state['features'][condition][f])
        predict_dict[f] = {'condition': np.repeat(condition, len(predict)),
                           'file': np.repeat(f, len(predict)),
                           'time': np.round(np.arange(0, len(predict) * 0.1, 0.1), 2),
                           'behavior': predict}
        predict_df = pd.DataFrame(predict_dict[f])
        labels = predict_df['behavior'].value_counts(sort=False).index
        file_id = np.repeat(predict_df['file'].value_counts(sort=False).index,
                            len(np.unique(labels)))
        values = predict_df['behavior'].value_counts(sort=False).values
        # summary dataframe
        df = pd.DataFrame()
        df['values'] = values
        df['file_id'] = file_id
        df['labels'] = labels
        duration_pie_df.append(df)
    concat_df = pd.concat([duration_pie_df[f] for f in range(len(duration_pie_df))])
    return convert_df(concat_df)




