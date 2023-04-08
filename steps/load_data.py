from utils.import_utils import *
from utils.visual_utils import *


def load_view():
    st.subheader('Machine learning classifier + new data upload')
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
            if 'classifier' not in st.session_state:
                st.markdown(f":blue[classifier] is in :orange[memory!]")
        except:
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
        try:
            conditions_list = list(st.session_state['features'].keys())
            st.markdown(f":blue[previously saved features] from conditions: "
                        f":orange[{' & '.join([i.rpartition('_')[2] for i in conditions_list])}!]")
        except:
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

        try:
            condition_plot()
        except:
            pass

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown(f'<span style="color:grey">LUPE X B-SOiD is developed by Alexander Hsu and '
                    f' Justin James </span>',
                    unsafe_allow_html=True)
