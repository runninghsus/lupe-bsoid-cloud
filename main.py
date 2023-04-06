import streamlit as st
import deeplabcut
import os
import io
import pathlib
import glob


HERE = pathlib.Path(__file__).parent.resolve()


st.set_page_config(
    page_title="LUPE X B-SOiD",
    layout="wide",
    menu_items={
    }
)

left_col, right_col = st.columns(2)
left_expand = left_col.expander('Select a video file:', expanded=True)
uploaded_file = left_expand.file_uploader('Upload video files',
                                          accept_multiple_files=False, type=['avi', 'mp4'], key='video')
temporary_location = False
if uploaded_file is not None:
    g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
    temporary_location = "./testout_simple.mp4"
    with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
        out.write(g.read())  ## Read bytes into file
    out.close()

if st.button('analyze pose'):
    config_path = os.path.join(HERE, 'bottomup_clear-hsu-2021-09-21/config.yaml')
    st.write(f'config file: {config_path}')
    with st.spinner('running deeplabcut...'):
        deeplabcut.analyze_videos(config_path, [temporary_location], save_as_csv=True)
if st.button('clear all'):
    for filename in glob.glob(str.join('', (str(HERE), '/testout_simple*'))):
        os.remove(filename)
