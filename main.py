import streamlit as st
from steps import menu, load_data, annotate
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


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


with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


name, authentication_status, username = authenticator.login('Login', 'main')
if authentication_status:
    with st.sidebar:
        selected = option_menu(None, ['home', 'data upload', 'annotate behaviors'],
                               icons=['house', 'file-earmark-arrow-up', 'pencil-square'],
                               menu_icon="cast", default_index=0, orientation="vertical",
                               styles={
                                   "container": {"padding": "0!important", "background-color": "#fafafa"},
                                   "icon": {"color": "black", "font-size": "20px"},
                                   "nav-link": {"color": "black", "font-size": "16px", "text-align": "center",
                                                "margin": "0px",
                                                "--hover-color": "#eee"},
                                   "nav-link-selected": {"font-size": "16px", "font-weight": "normal",
                                                         "color": "black", "background-color": "#FE589E"},
                               }
                               )
        _, midcol, _ = st.columns(3)
        with midcol:
            authenticator.logout('Logout', 'main')

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

elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
    # if st.button('New User Sign Up'):
# try:
#     if authenticator.register_user('Register user', preauthorization=False):
#         with open('./config.yaml', 'w') as file:
#             yaml.dump(config, file, default_flow_style=False)
#         st.success('User registered successfully')
# except Exception as e:
#     st.error(e)








