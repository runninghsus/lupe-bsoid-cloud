import streamlit as st
import deeplabcut
import os
import io

st.set_page_config(
    page_title="LUPE X B-SOiD",
    layout="wide",
    menu_items={
    }
)
# st.write(os.getcwd()) /home/user/app

left_col, right_col = st.columns(2)
left_expand = left_col.expander('Select a video file:', expanded=True)
uploaded_file = left_expand.file_uploader('Upload video files',
                                          accept_multiple_files=False, type=['avi', 'mp4'], key='video')
temporary_location = False

if uploaded_file is not None:
    g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
    temporary_location = "./testout_simple.mp4"

    with open(temporary_location[0], 'wb') as out:  ## Open temporary file as bytes
        out.write(g.read())  ## Read bytes into file
    out.close()
    config_path = r'/home/user/app/bottomup_clear-hsu-2021-09-21/config.yaml'
    deeplabcut.analyze_videos(config_path, temporary_location)
    # close file

# if st.button('analyze pose'):
#     st.write(f'video will be from {temporary_location}')
#     config_path = r'./bottomup_clear-hsu-2021-09-21/config.yaml'
#     st.write(f'config file: {config_path}')
#     # deeplabcut.analyze_videos(config_path, [temporary_location], save_as_csv=True)
