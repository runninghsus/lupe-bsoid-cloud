import streamlit
import time
from utils.import_utils import *
from utils.visual_utils import *
from utils.download_utils import *
from stqdm import stqdm


def load_view():
    st.markdown(f" <h1 style='text-align: left; color: #000000; font-size:30px; "
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
            print(st.session_state['classifier'])
            text_ = f":orange[reset classifier] in :blue[memory!]"
            def clear_classifier():
                del st.session_state['classifier']

            st.button(text_, on_click=clear_classifier)

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
            if 'classifier' in st.session_state:
                st.experimental_rerun()
        try:
            conditions_list = list(st.session_state['features'].keys())
            text_ = f":orange[reset data] from conditions: :blue[{' & '.join([i.rpartition('_')[2] for i in conditions_list])}!]"
            def clear_features():
                del st.session_state['features']

            st.button(text_, on_click=clear_features)

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
                    for i, condition in enumerate(
                            stqdm(list(uploaded_files.keys()),
                                  desc=f"Extracting spatiotemporal features from "
                                       f"{' & '.join([i.rpartition('_')[2] for i in conditions_list])}")):
                        loader = csv_upload(data_raw[i], pose_chosen, condition, framerate=30)
                        features[condition] = loader.main()
                    if 'features' not in st.session_state:
                        st.session_state['features'] = features
                    st.markdown(f":blue[saved features] from conditions: "
                                f":orange[{' & '.join([i.rpartition('_')[2] for i in conditions_list])}!]")
            except:
                pass
            if 'features' in st.session_state:
                st.experimental_rerun()
        try:
            condition_plot()
        except:
            pass

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown(f'<span style="color:grey; font-family:Avenir;">'
                    f'LUPE X B-SOiD is developed by Alexander Hsu and '
                    f' Justin James </span>',
                    unsafe_allow_html=True)
