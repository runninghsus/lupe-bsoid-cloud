import streamlit as st
import pandas as pd
import numpy as np
from utils.import_utils import read_csvfiles, get_bodyparts, csv_upload
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.colors as mcolors
import plotly.graph_objects as go


def load_model(model):
    loaded_model = joblib.load(model)
    return loaded_model


def rescale_classifier(feats_targets_file, training_file):
    [_, _, scalar, _] = joblib.load(feats_targets_file)
    [_, X_train, Y_train, _, _, _] = joblib.load(training_file)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                 criterion='gini',
                                 class_weight='balanced_subsample')
    X_train_inv = scalar.inverse_transform(X_train[-1][0])
    clf.fit(X_train_inv, Y_train[-1][0])
    return clf


def condition_prompt(uploaded_files, num_cond):
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
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                uploaded_files[f'condition_{row * 2 + 1}'] = right_expander.file_uploader(
                    'Upload corresponding pose csv files',
                    accept_multiple_files=True,
                    type='csv',
                    key=f'pose_upload_2_{row}')
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            uploaded_files[f'condition_{row * 2 + 1}'] = right_expander.file_uploader(
                'Upload corresponding pose csv files',
                accept_multiple_files=True,
                type='csv',
                key=f'pose_upload_2_{row}')


def condition_plot():
    behavior_classes = st.session_state['classifier'].classes_
    option_expander = st.expander("Configure Plot",
                                  expanded=False)
    behavior_colors = []
    all_c_options = list(mcolors.CSS4_COLORS.keys())

    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]

    for i, class_id in enumerate(behavior_classes):
        behavior_colors.append(option_expander.selectbox(f'Color for {behavior_classes[i]}',
                                                         all_c_options,
                                                         index=all_c_options.index(default_colors[i]),
                                                         key=f'color_option{i}'))
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = st.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        pie_predict(left_expander,
                    list(st.session_state['features'].keys())[count],
                    behavior_colors)
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                pie_predict(right_expander,
                            list(st.session_state['features'].keys())[count],
                            behavior_colors)
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            pie_predict(right_expander,
                        list(st.session_state['features'].keys())[count],
                        behavior_colors)
            count += 1


def pie_predict(placeholder, condition, behavior_colors):
    predict = st.session_state['classifier'].predict(st.session_state['features'][condition][0])
    predict_dict = {'condition': np.repeat(condition, len(predict)),
                    'behavior': predict}
    df = pd.DataFrame(data=predict_dict)
    behavior_classes = np.unique(predict)
    with placeholder:
        labels = df['behavior'].value_counts().index
        values = df['behavior'].value_counts().values
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=16,
                          marker=dict(colors=behavior_colors,
                                      line=dict(color='#000000', width=1)))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def load_view():
    st.subheader('Machine learning classifier + new data upload')
    bsoid_t, asoid_t = st.tabs(['B-SOiD', 'A-SOiD'])
    with bsoid_t:
        uploaded_model = st.file_uploader('Upload your ',
                                          accept_multiple_files=False,
                                          type='pkl',
                                          key=f'bsoid_model_pkl')
        try:
            clf = load_model(uploaded_model)
        except:
            st.warning('please upload pkl file')
    with asoid_t:
        if st.session_state['classifier'] is not None:
            st.markdown(f":blue[classifier] is in :orange[memory!]")
        else:
            file1, file2 = st.columns(2)
            feats_targets_file = file1.file_uploader('Upload your feats_targets.sav',
                                                     accept_multiple_files=False,
                                                     type='sav',
                                                     key=f'asoid_featstargets_pkl')
            training_file = file2.file_uploader('Upload your iterX.sav',
                                                accept_multiple_files=False,
                                                type='sav',
                                                key=f'asoid_training_pkl')

            try:
                if 'classifier' not in st.session_state:
                    st.session_state['classifier'] = rescale_classifier(feats_targets_file, training_file)
            except:
                st.warning('please upload sav file')

        if st.session_state['features'] is not None:
            conditions_list = list(st.session_state['features'].keys())
            st.markdown(f":blue[previously saved features] from conditions: "
                        f":orange[{' & '.join([i.rpartition('_')[2] for i in conditions_list])}!]")
        else:

            num_cond = st.number_input('How many conditions?', min_value=2, max_value=10, value=2)
            uploaded_files = {f'condition_{key}': [] for key in range(num_cond)}
            features = {f'condition_{key}': [] for key in range(num_cond)}
            condition_prompt(uploaded_files, num_cond)

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
                    if 'features' not in st.session_state:
                        st.session_state['features'] = features
                    st.markdown(f":blue[saved features] from conditions: "
                                f":orange[{' & '.join([i.rpartition('_')[2] for i in conditions_list])}!]")
            except:
                pass

        condition_plot()

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown(f'<span style="color:grey">LUPE X B-SOiD is developed by Alexander Hsu and '
                    f' Justin James </span>',
                    unsafe_allow_html=True)
