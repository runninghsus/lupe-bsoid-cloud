import streamlit as st
from steps import menu, load_data, annotate
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide",
                   page_title='LUPE X B-SOiD',
                   page_icon='🪤')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
_, center, _ = st.columns([1, 12, 0.1])
# st.image('./images/logo.png')
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
        min-width: 300px;
        max-width: 300px;   
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    selected = option_menu(None, ['home', 'data upload', 'annotate behaviors'],
                           icons=['house', 'file-earmark-arrow-up', 'pencil-square'],
                           menu_icon="cast", default_index=0, orientation="vertical",
                           styles={
                               "container": {"padding": "0!important", "background-color": "#fafafa"},
                               "icon": {"color": "black", "font-size": "20px"},
                               "nav-link": {"color": "black", "font-size": "16px", "text-align": "center", "margin": "0px",
                                            "--hover-color": "#eee"},
                               "nav-link-selected": {"font-size": "16px", "font-weight": "normal",
                                                     "color": "black", "background-color":"#FE589E"},
                           }
                           )


def navigation():
    if selected == 'home':
        menu.load_view()
    elif selected == 'data upload':
        load_data.load_view()
    elif selected == 'annotate behaviors':
        annotate.load_view()
    elif selected == None:
        menu.load_view()


navigation()