import streamlit as st
import os
import io
from utils.import_utils import read_csvfiles, behavior_movies

def load_view():
    st.header('Annotate Behaviors')
    left_col, right_col = st.columns(2)

    # left stays
    left_expander = left_col.expander(f'Upload video to show example video:',
                                      expanded=True)
    right_expander = right_col.expander(f'Upload corresponding pose file to predict behavior:',
                                      expanded=True)
    example_video = left_expander.file_uploader('Upload video files',
                                                accept_multiple_files=False,
                                                type=['avi', 'mp4'], key='video')
    example_pose = right_expander.file_uploader('Upload corresponding pose csv files',
                                                    accept_multiple_files=False,
                                                    type='csv', key='pose')

    # try:
    example_data = read_csvfiles([example_pose])
    # copy video to local, only when csv and mp4/avi are uploaded
    if example_video is not None and len(example_data) > 0:
        if os.path.exists(example_video.name):
            temporary_location = f'{example_video.name}'
        else:
            g = io.BytesIO(example_video.read())  # BytesIO Object
            temporary_location = f'{example_video.name}'
            with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            out.close()

    movie_maker = behavior_movies(temporary_location, example_data, framerate=30)
    movie_maker.main()




    # except:
    #     pass


    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown(f'<span style="color:grey">LUPE X B-SOiD is developed by Alexander Hsu and '
                    f' Justin James </span>',
                    unsafe_allow_html=True)
