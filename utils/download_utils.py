import streamlit as st
import pandas as pd
import numpy as np
from utils.feature_utils import get_avg_kinematics


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


def get_num_bouts(predict, behavior_classes):
    bout_counts = []
    bout_start_idx = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
    bout_start_label = predict[bout_start_idx]
    for b in behavior_classes:
        idx_b = np.where(bout_start_label == int(b))[0]
        if len(idx_b) > 0:
            bout_counts.append(len(idx_b))
        else:
            bout_counts.append(0)
    return bout_counts


def bout_bar_csv(condition):
    predict_dict = {key: [] for key in range(len(st.session_state['features'][condition]))}
    bout_counts = {key: [] for key in range(len(st.session_state['features'][condition]))}
    behavior_classes = st.session_state['classifier'].classes_
    predict = []
    bout_counts_df = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    for f in range(len(predict)):
        bout_counts[f] = get_num_bouts(predict[f], behavior_classes)
        predict_dict[f] = {'condition': np.repeat(condition, len(behavior_classes)),
                           'file': np.repeat(f, len(behavior_classes)),
                           'behavior': behavior_classes,
                           'number of bouts': bout_counts[f],
                           }
        bout_counts_df.append(pd.DataFrame(predict_dict[f]))
    concat_df = pd.concat([bout_counts_df[f] for f in range(len(bout_counts_df))])
    return convert_df(concat_df)


def get_duration_bouts(predict, behavior_classes, framerate=10):
    behav_durations = []
    bout_start_idx = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
    bout_durations = np.hstack([np.diff(bout_start_idx), len(predict) - np.max(bout_start_idx)])
    bout_start_label = predict[bout_start_idx]
    for b in behavior_classes:
        idx_b = np.where(bout_start_label == int(b))[0]
        if len(idx_b) > 0:
            behav_durations.append(bout_durations[idx_b]/framerate)
        else:
            behav_durations.append(0)
    return behav_durations


def duration_ridge_csv(condition):
    predict_dict = {key: [] for key in range(len(st.session_state['features'][condition]))}
    durations_ = {key: [] for key in range(len(st.session_state['features'][condition]))}
    behavior_classes = st.session_state['classifier'].classes_
    predict = []
    durations_df = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    for f in range(len(predict)):
        durations_[f] = get_duration_bouts(predict[f], behavior_classes)
        predict_dict[f] = {'condition': np.hstack([np.repeat(condition, len(durations_[f][i]))
                                                   for i in range(len(durations_[f]))]),
                           'file': np.hstack([np.repeat(f, len(durations_[f][i]))
                                              for i in range(len(durations_[f]))]),
                           'behavior': np.hstack([np.repeat(behavior_classes[i],
                                                  len(durations_[f][i])) for i in range(len(durations_[f]))]),
                           'duration': np.hstack(durations_[f]),
                           }
        durations_df.append(pd.DataFrame(predict_dict[f]))
    concat_df = pd.concat([durations_df[f] for f in range(len(durations_df))])
    return convert_df(concat_df)


def get_transitions(predict, behavior_classes):
    class_int = [int(i) for i in behavior_classes]
    tm = [[0] * np.unique(class_int) for _ in np.unique(class_int)]
    for (i, j) in zip(predict, predict[1:]):
        tm[int(i)][int(j)] += 1
    tm_df = pd.DataFrame(tm)
    tm_array = np.array(tm)
    tm_norm = tm_array / tm_array.sum(axis=1)
    return tm_array, tm_norm


def transmat_csv(condition):
    transitions_ = []
    behavior_classes = st.session_state['classifier'].classes_
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    for f in range(len(predict)):
        count_tm, prob_tm = get_transitions(predict[f], behavior_classes)
        transitions_.append(prob_tm)
    mean_transitions = np.mean(transitions_, axis=0)
    transmat_df = pd.DataFrame(mean_transitions)
    return convert_df(transmat_df)


def kinematics_csv(condition, bp_selects):

    predict_dict = {key: [] for key in range(len(st.session_state['features'][condition]))}
    behavior_classes = st.session_state['classifier'].classes_
    names = [f'behavior {int(key)}' for key in behavior_classes]
    pose = st.session_state['pose'][condition]
    kinematics_df = []
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
        bout_disp_bps = []
        bout_duration_bps = []
        bout_avg_speed_bps = []
        for bp_select in bp_selects:
            bodypart = st.session_state['bodypart_names'].index(bp_select)
            bout_disp_all = []
            bout_duration_all = []
            bout_avg_speed_all = []
            for file_chosen in range(len(predict)):
                behavior, behavioral_start_time, behavior_duration, bout_disp, bout_duration, bout_avg_speed = \
                    get_avg_kinematics(predict[file_chosen], pose[file_chosen], bodypart, framerate=10)
                bout_disp_all.append(bout_disp)
                bout_duration_all.append(bout_duration)
                bout_avg_speed_all.append(bout_avg_speed)
            bout_disp_bps.append(bout_disp_all)
            bout_duration_bps.append(bout_duration_all)
            bout_avg_speed_bps.append(bout_avg_speed_all)
        # TODO: create dictionary with kineamtics
        # predict_dict[f] = {'condition': np.hstack([np.repeat(condition, len(durations_[f][i]))
        #                                            for i in range(len(durations_[f]))]),
        #                    'file': np.hstack([np.repeat(f, len(durations_[f][i]))
        #                                       for i in range(len(durations_[f]))]),
        #                    'behavior': np.hstack([np.repeat(behavior_classes[i],
        #                                           len(durations_[f][i])) for i in range(len(durations_[f]))]),
        #                    'distance': np.hstack(bout_disp_bps),
        #                    'duration':
        #                    'duration':
        #                    }
        kinematics_df.append(pd.DataFrame(predict_dict[f]))
    concat_df = pd.concat([kinematics_df[f] for f in range(len(kinematics_df))])
    return convert_df(concat_df)
