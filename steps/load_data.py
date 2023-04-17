import stqdm
import streamlit
import time
from utils.import_utils import *
from utils.visual_utils import *
from utils.download_utils import *
from stqdm import stqdm


def load_view():
    st.markdown(f" <h1 style='text-align: left; color: #FF6A95; font-size:30px; "
                f"font-family:Avenir; font-weight:normal;'>Machine learning classifier + new data upload</h1> "
                , unsafe_allow_html=True)
    st.write('')
    bsoid_t, asoid_t = st.tabs(['B-SOiD', 'A-SOiD'])
    with bsoid_t:
        uploaded_model = st.file_uploader('Upload your ',
                                          accept_multiple_files=False,
                                          type='pkl',
                                          key=f'bsoid_model_pkl')
        try:
            clf = load_pickle_model(uploaded_model)
        except:
            st.warning('please upload pkl file')
    with asoid_t:
        try:
            # print(st.session_state['classifier'])
            if 'classifier' in st.session_state:
                text_ = f":red[**RESET**] classifier in memory!"

            def clear_classifier():
                del st.session_state['classifier']

            st.button(text_, on_click=clear_classifier)

        except:
            sex_selection = st.radio('select sex', ['male', 'female'])
            if sex_selection == 'male':
                st.markdown('loaded :blue[Male] model')
            elif sex_selection == 'female':
                st.markdown('loaded :orange[Female] model')

            if st.checkbox('use demo model and data'):
                with open('./models/demo.pkl', 'rb') as fr:
                    st.session_state['classifier'] = pickle.load(fr)
                with open('./data/demo_feats.pkl', 'rb') as fr:
                    st.session_state['features'] = pickle.load(fr)
                with open('./data/demo_pose.pkl', 'rb') as fr:
                    st.session_state['pose'] = pickle.load(fr)
                with open('./data/demo_bodypart_names.pkl', 'rb') as fr:
                    st.session_state['bodypart_names'] = pickle.load(fr)
                with open('./data/demo_bodypart_idx.pkl', 'rb') as fr:
                    st.session_state['bodypart_idx'] = pickle.load(fr)
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
                    # pickle.dump(st.session_state['classifier'], open('./models/demo.pkl', 'wb'))
            except:
                st.warning('please upload sav file')
            if 'classifier' in st.session_state:
                st.experimental_rerun()
        try:
            conditions_list = list(st.session_state['features'].keys())
            text_ = f":red[**RESET**] data from conditions: {' & '.join([i.rpartition('_')[2] for i in conditions_list])}!"

            def clear_data():
                del st.session_state['features']
                del st.session_state['pose']
                del st.session_state['bodypart_names']
                del st.session_state['bodypart_idx']

            st.button(text_, on_click=clear_data)

        except:
            num_cond = st.number_input('How many conditions?', min_value=2, max_value=10, value=2)
            uploaded_files = {f'condition_{key}': [] for key in range(num_cond)}
            pose = {f'condition_{key}': [] for key in range(num_cond)}
            features = {f'condition_{key}': [] for key in range(num_cond)}
            condition_prompt(uploaded_files, num_cond)
            try:
                data_raw = []
                for i, condition in enumerate(list(uploaded_files.keys())):
                    placeholder = st.empty()
                    data_raw.append(read_csvfiles(uploaded_files[condition]))
                    if i == 0:
                        p, pose_chosen = get_bodyparts(placeholder, data_raw[i])
                        if 'bodypart_names' not in st.session_state or 'bodypart' not in st.session_state:
                            st.session_state['bodypart_names'] = p
                            st.session_state['bodypart_idx'] = pose_chosen
                conditions_list = list(uploaded_files.keys())
                if st.button(f"extract features from conditions: "
                             f"{' & '.join([i.rpartition('_')[2] for i in conditions_list])}"):
                    for i, condition in enumerate(
                            stqdm(list(uploaded_files.keys()),
                                  desc=f"Extracting spatiotemporal features from "
                                       f"{' & '.join([i.rpartition('_')[2] for i in conditions_list])}")):
                        loader = csv_upload(data_raw[i], pose_chosen, condition, framerate=30)
                        pose[condition], features[condition] = loader.main()
                    if 'pose' not in st.session_state:
                        st.session_state['pose'] = pose
                    if 'features' not in st.session_state:
                        st.session_state['features'] = features
                    st.markdown(f":blue[saved features] from conditions: "
                                f":orange[{' & '.join([i.rpartition('_')[2] for i in conditions_list])}!]")
                    # pickle.dump(st.session_state['bodypart_names'], open('./data/demo_bodypart_names.pkl', 'wb'))
                    # pickle.dump(st.session_state['bodypart_idx'], open('./data/demo_bodypart_idx.pkl', 'wb'))
                    # pickle.dump(st.session_state['pose'], open('./data/demo_pose.pkl', 'wb'))
                    # pickle.dump(st.session_state['features'], open('./data/demo_feats.pkl', 'wb'))
            except:
                pass
            if 'features' in st.session_state:
                st.experimental_rerun()
        st.write('---')
        # condition_kinematix_plot()
        try:
            if 'pose' in st.session_state:
                mid_expander = st.expander('Analysis method', expanded=True)
                analysis_chosen = mid_expander.radio('',
                                           ['ethogram', 'behavioral location', 'behavioral ratio', 'frequency',
                                            'duration', 'transition', 'kinematics'],
                                           horizontal=True)
                if analysis_chosen == 'ethogram':
                    condition_etho_plot()
                if analysis_chosen == 'behavioral ratio':
                    condition_pie_plot()
                if analysis_chosen == 'behavioral location':
                    st.write('placeholder for behavioral location')
                if analysis_chosen == 'frequency':
                    condition_bar_plot()
                if analysis_chosen == 'duration':
                    condition_ridge_plot()
                if analysis_chosen == 'transition':
                    condition_transmat_plot()
                if analysis_chosen == 'kinematics':
                    condition_kinematix_plot()
                # st.write('placeholder for pose kinematics')
        except:
            pass

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"LUPE B-SOiD is developed by Alexander Hsu and Justin James</h1> "
                    , unsafe_allow_html=True)
