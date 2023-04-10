import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from utils.download_utils import *

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def ethogram_plot(condition, new_predictions, behavior_names, behavior_colors, length_):
    colL, colR = st.columns(2)
    if len(new_predictions) == 1:
        colL.markdown(':orange[1] file only')
        f_select = 0
    else:
        f_select = colL.slider('select file to generate ethogram',
                             min_value=1, max_value=len(new_predictions), value=1)
    file_idx = f_select - 1
    prefill_array = np.zeros((len(new_predictions[file_idx]),
                              len(st.session_state['classifier'].classes_)))
    default_colors_wht = ['w']
    default_colors_wht.extend(behavior_colors)
    cmap_ = ListedColormap(default_colors_wht)


    count = 0
    for b in np.unique(st.session_state['classifier'].classes_):
        idx_b = np.where(new_predictions[file_idx] == b)[0]
        prefill_array[idx_b, count] = b + 1
        count += 1
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    seed_num = colR.number_input('seed for segment',
                                 min_value=0, max_value=None, value=42,
                                 key=f'cond{condition}_seed')
    np.random.seed(seed_num)
    behaviors_with_names = behavior_names
    if colL.checkbox('use randomized time',
                   value=True,
                   key=f'cond{condition}_ckbx'):
        rand_start = np.random.choice(prefill_array.shape[0] - length_, 1, replace=False)
        ax.imshow(prefill_array[int(rand_start):int(rand_start + length_), :].T, cmap=cmap_)
        # ax.set_yticks(np.arange(0, len(behaviors_with_names), 1))
        # ax.set_yticklabels(np.arange(0, len(behaviors_with_names), 1))
        ax.set_xticks(np.arange(0, length_, int(length_ / 5)))
        ax.set_xticklabels(np.arange(int(rand_start), int(rand_start + length_), int(length_ / 5)))
        ax.set_xlabel('Frame #')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        rand_start = 0
        ax.imshow(prefill_array[rand_start:rand_start + length_, :].T, cmap=cmap_)
        ax.set_xticks(np.arange(rand_start, length_, int(length_ / 1)))
        ax.set_xticklabels(np.arange(0, length_, int(length_ / 1)))
        ax.set_xlabel('Frame #')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, prefill_array, rand_start


def ethogram_predict(placeholder, condition, behavior_colors, length_):
    behavior_classes = st.session_state['classifier'].classes_
    predict = []
    # TODO: find a color workaround if a class is missing
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        etho_placeholder = st.empty()
        fig, prefill_array, rand_start = ethogram_plot(condition, predict, behavior_classes,
                                                       list(behavior_colors.values()), length_)
        etho_placeholder.pyplot(fig)


def pie_predict(placeholder, condition, behavior_colors):
    predict = []
    # TODO: find a color workaround if a class is missing
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    predict_dict = {'condition': np.repeat(condition, len(np.hstack(predict))),
                    'behavior': np.hstack(predict)}
    df_raw = pd.DataFrame(data=predict_dict)
    labels = df_raw['behavior'].value_counts(sort=False).index
    values = df_raw['behavior'].value_counts(sort=False).values
    # summary dataframe
    df = pd.DataFrame()
    df["values"] = values
    df['labels'] = labels
    df["colors"] = df["labels"].apply(lambda x: behavior_colors.get(x))  # to connect Column value to Color in Dict
    with placeholder:
        fig = go.Figure(data=[go.Pie(labels=df["labels"], values=df["values"], hole=.4)])
        fig.update_traces(hoverinfo='label+percent',
                          textinfo='value',
                          textfont_size=16,
                          marker=dict(colors=df["colors"],
                                      line=dict(color='#000000', width=1)))
        st.plotly_chart(fig, use_container_width=True)


def condition_etho_plot():
    behavior_classes = st.session_state['classifier'].classes_
    option_expander = st.expander("Configure Plot",
                                  expanded=False)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    for i, class_id in enumerate(behavior_classes):
        behavior_colors[class_id] = option_expander.selectbox(f'Color for {behavior_classes[i]}',
                                                              all_c_options,
                                                              index=all_c_options.index(default_colors[i]),
                                                              key=f'color_option{i}')
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    length_ = st.slider('number of frames',
                        min_value=25, max_value=250,
                        value=100,
                        key=f'length_slider')
    for row in range(rows):
        left_col, right_col = st.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        ethogram_predict(left_expander,
                         list(st.session_state['features'].keys())[count],
                         behavior_colors, length_)
        predict_csv = csv_predict(
            list(st.session_state['features'].keys())[count],
        )

        left_expander.download_button(
            label="Download data as CSV",
            data=predict_csv,
            file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                ethogram_predict(right_expander,
                                 list(st.session_state['features'].keys())[count],
                                 behavior_colors, length_)
                predict_csv = csv_predict(
                    list(st.session_state['features'].keys())[count],
                )

                right_expander.download_button(
                    label="Download data as CSV",
                    data=predict_csv,
                    file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            ethogram_predict(right_expander,
                             list(st.session_state['features'].keys())[count],
                             behavior_colors, length_)
            predict_csv = csv_predict(
                list(st.session_state['features'].keys())[count],
            )

            right_expander.download_button(
                label="Download data as CSV",
                data=predict_csv,
                file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def condition_pie_plot():
    behavior_classes = st.session_state['classifier'].classes_
    option_expander = st.expander("Configure Plot",
                                  expanded=False)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    for i, class_id in enumerate(behavior_classes):
        behavior_colors[class_id] = option_expander.selectbox(f'Color for {behavior_classes[i]}',
                                                              all_c_options,
                                                              index=all_c_options.index(default_colors[i]),
                                                              key=f'color_option{i}')
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
        # predict_csv = csv_predict(
        #     list(st.session_state['features'].keys())[count],
        # )
        predict_csv = duration_pie_csv(
            list(st.session_state['features'].keys())[count],
        )
        left_expander.download_button(
            label="Download data as CSV",
            data=predict_csv,
            file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                pie_predict(right_expander,
                            list(st.session_state['features'].keys())[count],
                            behavior_colors)
                # predict_csv = csv_predict(
                #     list(st.session_state['features'].keys())[count],
                # )
                predict_csv = duration_pie_csv(
                    list(st.session_state['features'].keys())[count],
                )
                right_expander.download_button(
                    label="Download data as CSV",
                    data=predict_csv,
                    file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            pie_predict(right_expander,
                        list(st.session_state['features'].keys())[count],
                        behavior_colors)
            # predict_csv = csv_predict(
            #     list(st.session_state['features'].keys())[count],
            # )
            predict_csv = duration_pie_csv(
                list(st.session_state['features'].keys())[count],
            )
            right_expander.download_button(
                label="Download data as CSV",
                data=predict_csv,
                file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1
