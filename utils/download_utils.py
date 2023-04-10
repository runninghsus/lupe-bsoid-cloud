import streamlit as st
import pandas as pd
import numpy as np


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


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
