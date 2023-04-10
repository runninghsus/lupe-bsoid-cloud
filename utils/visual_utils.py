import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from utils.download_utils import *
import plotly.express as px

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
                               min_value=1, max_value=len(new_predictions), value=1,
                               key=f'ethogram_slider_{condition}')
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
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        etho_placeholder = st.empty()
        fig, prefill_array, rand_start = ethogram_plot(condition, predict, behavior_classes,
                                                       list(behavior_colors.values()), length_)
        etho_placeholder.pyplot(fig)


def condition_etho_plot():
    behavior_classes = st.session_state['classifier'].classes_
    length_container = st.container()
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    length_ = length_container.slider('number of frames',
                                      min_value=25, max_value=250,
                                      value=100,
                                      key=f'length_slider')
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
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


def pie_predict(placeholder, condition, behavior_colors):
    behavior_classes = st.session_state['classifier'].classes_
    # behavior_classes = np.arange(7)
    predict = []
    # TODO: find a color workaround if a class is missing
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    predict_dict = {'condition': np.repeat(condition, len(np.hstack(predict))),
                    'behavior': np.hstack(predict)}
    df_raw = pd.DataFrame(data=predict_dict)
    labels = df_raw['behavior'].value_counts(sort=False).index
    values = df_raw['behavior'].value_counts(sort=False).values
    names = [f'behavior {int(key)}' for key in behavior_classes]
    # summary dataframe
    df = pd.DataFrame()
    # do i need this?
    behavior_labels = []
    for l in labels:
        behavior_labels.append(behavior_classes[int(l)])
    df["values"] = values
    df['labels'] = behavior_labels
    df["colors"] = df["labels"].apply(lambda x:
                                      behavior_colors.get(x))  # to connect Column value to Color in Dict
    with placeholder:
        fig = go.Figure(data=[go.Pie(labels=[names[int(i)] for i in df["labels"]], values=df["values"], hole=.4)])
        fig.update_traces(hoverinfo='label+percent',
                          textinfo='value',
                          textfont_size=16,
                          marker=dict(colors=df["colors"],
                                      line=dict(color='#000000', width=1)))
        st.plotly_chart(fig, use_container_width=True)


def condition_pie_plot():
    behavior_classes = st.session_state['classifier'].classes_
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        pie_predict(left_expander,
                    list(st.session_state['features'].keys())[count],
                    behavior_colors)
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


def bar_predict(placeholder, condition, behavior_colors):
    behavior_classes = st.session_state['classifier'].classes_
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        bar_placeholder = st.empty()
        bout_counts = []
        for file_idx in range(len(predict)):
            bout_counts.append(get_num_bouts(predict[file_idx], behavior_classes))
        bout_mean = np.mean(bout_counts, axis=0)
        bout_std = np.std(bout_counts, axis=0)
        names = [f'behavior {int(key)}' for key in behavior_classes]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            # name=f'{condition}',
            x=names, y=bout_mean,
            error_y=dict(type='data', array=bout_std),
            width=0.5,
            marker_color=pd.Series(behavior_colors))
        )
        fig.update_layout()
        bar_placeholder.plotly_chart(fig, use_container_width=True)


def condition_bar_plot():
    behavior_classes = st.session_state['classifier'].classes_
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        bar_predict(left_expander,
                    list(st.session_state['features'].keys())[count],
                    behavior_colors)
        bar_csv = bout_bar_csv(
            list(st.session_state['features'].keys())[count],
        )
        left_expander.download_button(
            label="Download data as CSV",
            data=bar_csv,
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
                bar_predict(right_expander,
                            list(st.session_state['features'].keys())[count],
                            behavior_colors)
                bar_csv = bout_bar_csv(
                    list(st.session_state['features'].keys())[count],
                )
                right_expander.download_button(
                    label="Download data as CSV",
                    data=bar_csv,
                    file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            bar_predict(right_expander,
                        list(st.session_state['features'].keys())[count],
                        behavior_colors)
            bar_csv = bout_bar_csv(
                list(st.session_state['features'].keys())[count],
            )
            right_expander.download_button(
                label="Download data as CSV",
                data=bar_csv,
                file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def ridge_predict(placeholder, condition, behavior_colors):
    behavior_classes = st.session_state['classifier'].classes_
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:

        ridge_placeholder = st.empty()
        colL, colR = st.columns(2)
        if len(predict) == 1:
            colL.markdown(':orange[1] file only')
            f_select = 0
        else:
            f_select = colL.slider('select file to generate ridge plot',
                                   min_value=1, max_value=len(predict), value=1,
                                   key=f'ridge_slider_{condition}')
        file_chosen = f_select - 1
        duration_ = []
        for file_idx in range(len(predict)):
            duration_.append(get_duration_bouts(predict[file_idx], behavior_classes))
        colors = [mcolors.to_hex(i) for i in list(behavior_colors.values())]
        duration_matrix = boolean_indexing(duration_[file_chosen])
        max_dur = colR.slider('max duration',
                              min_value=0, max_value=int(np.nanmax(np.array(duration_matrix))),
                              value=int(np.nanmax(np.array(duration_matrix) * 0.7)),
                              key=f'maxdur_slider_{condition}')

        fig = go.Figure()
        names = [f'behavior {int(key)}' for key in behavior_classes]
        for data_line, color, name in zip(duration_matrix, colors, names):
            fig.add_trace(go.Violin(x=data_line, line_color=color, name=name))
        fig.update_traces(
            orientation='h', side='positive', width=3, points=False,
        )
        fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False, xaxis_range=[0, max_dur])
        ridge_placeholder.plotly_chart(fig, use_container_width=True)


def condition_ridge_plot():
    behavior_classes = st.session_state['classifier'].classes_
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        ridge_predict(left_expander,
                      list(st.session_state['features'].keys())[count],
                      behavior_colors)
        ridge_csv = duration_ridge_csv(
            list(st.session_state['features'].keys())[count],
        )
        left_expander.download_button(
            label="Download data as CSV",
            data=ridge_csv,
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
                ridge_predict(right_expander,
                              list(st.session_state['features'].keys())[count],
                              behavior_colors)
                ridge_csv = duration_ridge_csv(
                    list(st.session_state['features'].keys())[count],
                )
                right_expander.download_button(
                    label="Download data as CSV",
                    data=ridge_csv,
                    file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            ridge_predict(right_expander,
                          list(st.session_state['features'].keys())[count],
                          behavior_colors)
            ridge_csv = duration_ridge_csv(
                list(st.session_state['features'].keys())[count],
            )
            right_expander.download_button(
                label="Download data as CSV",
                data=ridge_csv,
                file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1
