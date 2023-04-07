import streamlit as st
import pandas as pd
import numpy as np
from utils.import_utils import read_csvfiles, get_bodyparts, csv_upload


def load_view():
    st.header('Data Upload')
    num_cond = st.number_input('How many conditions?', min_value=2, max_value=10, value=2)
    uploaded_files = {f'condition_{key}': [] for key in range(num_cond)}
    features = {f'condition_{key}': [] for key in range(num_cond)}
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    for row in range(rows):
        left_col, right_col = st.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        uploaded_files[f'condition_{row * 2}'] = left_expander.file_uploader('Upload corresponding pose csv files',
                                                       accept_multiple_files=True,
                                                       type='csv',
                                                       key=f'pose_upload_1_{row}')
        # right only when multiples of 2 or
        if row == rows-1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                uploaded_files[f'condition_{row * 2 + 1}'] = right_expander.file_uploader('Upload corresponding pose csv files',
                                                                accept_multiple_files=True,
                                                                type='csv',
                                                                key=f'pose_upload_2_{row}')
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            uploaded_files[f'condition_{row * 2 + 1}'] = right_expander.file_uploader('Upload corresponding pose csv files',
                                                            accept_multiple_files=True,
                                                            type='csv',
                                                            key=f'pose_upload_2_{row}')
    try:
        data_raw = []
        for i, condition in enumerate(list(uploaded_files.keys())):
            placeholder = st.empty()
            data_raw.append(read_csvfiles(uploaded_files[condition]))
            if i == 0:
                pose_chosen = get_bodyparts(placeholder, data_raw[i])
        conditions_list = list(uploaded_files.keys())

        if st.button(f"extract features from conditions: "
                     f"{' & '.join([i.rpartition('_')[2] for i in conditions_list])}"):
            for i, condition in enumerate(list(uploaded_files.keys())):
                loader = csv_upload(data_raw[i], pose_chosen, condition, framerate=30)
                features[condition] = loader.main()
        st.success(f"saved features from conditions: "
                   f"{' & '.join([i.rpartition('_')[2] for i in conditions_list])}!")
    except:
        pass



    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown(f'<span style="color:grey">LUPE X B-SOiD is developed by Alexander Hsu and '
                    f' Justin James </span>',
                    unsafe_allow_html=True)
