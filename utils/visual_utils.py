import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from utils.download_utils import *

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def ethogram_plot(new_predictions, behavior_names, behavior_colors, file_idx):
    prefill_array = np.zeros((len(new_predictions[file_idx]),
                              len(st.session_state['classifier'].classes_)))
    default_colors_wht = ['w']
    default_colors_wht.extend(behavior_colors)
    cmap_ = ListedColormap(default_colors_wht)
    colL, colR = st.columns(2)

    length_ = colL.slider('number of frames',
                          min_value=30, max_value=int(len(new_predictions[file_idx])/10),
                          value=100,
                          key=f'slider_{file_idx}')
    count = 0
    for b in np.unique(st.session_state['classifier'].classes_):
        idx_b = np.where(new_predictions[file_idx] == b)[0]
        prefill_array[idx_b, count] = b + 1
        count += 1
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    seed_num = colR.number_input('seed for segment',
                                 min_value=0, max_value=None, value=42,
                                 key=f'seed_{file_idx}')
    np.random.seed(seed_num)
    behaviors_with_names = behavior_names

    if st.checkbox('use randomized time',
                   value=True,
                   key=f'ckbx_{file_idx}'):
        rand_start = np.random.choice(prefill_array.shape[0] - length_, 1, replace=False)
        ax.imshow(prefill_array[int(rand_start):int(rand_start + length_), :].T, cmap=cmap_)
        # ax.set_yticks(np.arange(0, len(behaviors_with_names), 1))
        # ax.set_yticklabels(np.arange(0, len(behaviors_with_names), 1))
        ax.set_xticks(np.arange(0, length_, int(length_/5)))
        ax.set_xticklabels(np.arange(int(rand_start), int(rand_start + length_), int(length_/5)))
        ax.set_xlabel('Frame #')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        rand_start = 0
        ax.imshow(prefill_array[rand_start:rand_start+length_, :].T, cmap=cmap_)
        ax.set_xticks(np.arange(rand_start, length_, int(length_/1)))
        ax.set_xticklabels(np.arange(0, length_, int(length_/1)))
        ax.set_xlabel('Frame #')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, prefill_array, rand_start, length_


def ethogram_predict(placeholder, condition, behavior_colors):
    behavior_classes = st.session_state['classifier'].classes_
    predict = []
    # TODO: find a color workaround if a class is missing
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        fig, prefill_array, rand_start, length_ = ethogram_plot(predict, behavior_classes, list(behavior_colors.values()), f)
        st.pyplot(fig)



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
    for row in range(rows):
        left_col, right_col = st.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        ethogram_predict(left_expander,
                    list(st.session_state['features'].keys())[count],
                    behavior_colors)
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
                            behavior_colors)
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
                        behavior_colors)
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
