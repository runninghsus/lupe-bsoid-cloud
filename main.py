import streamlit as st
from steps import menu, load_data, annotate
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from PIL import Image
from pathlib import Path
import base64


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, width=500):
    img_html = f"<img src='data:image/png;base64,{img_to_bytes(img_path)}'  width='{width}px', class='img-fluid'>"
    return img_html


icon = './images/icon_only/color_transparent.png'
banner = './images/png/color_transparent_banner.png'

icon_img = Image.open(icon)
st.set_page_config(layout="wide",
                   page_title='LUPE X B-SOiD',
                   page_icon=icon_img)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
_, center, _ = st.columns([1, 12, 0.1])

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
        min-width: 250px;
        max-width: 250px;   
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -250px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)

with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

logo_placeholder = st.empty()
st.write('')
st.write('')
_, mid, _ = st.columns([1, 4, 1])
with mid:
    name, authentication_status, username = authenticator.login('Login', 'main')
if not authentication_status:
    logo_placeholder.markdown("<p style='text-align: center; color: grey; '>" + img_to_html(banner) + "</p>",
                              unsafe_allow_html=True)
elif authentication_status:
    with st.sidebar:
        st.markdown("<p style='text-align: center; color: grey; '>" + img_to_html(banner, width=200) + "</p>",
                    unsafe_allow_html=True)
        if 'user' not in st.session_state:
            st.session_state.user = username
        st.markdown(f" <h1 style='text-align: center; color: #FF6A95; font-size:18px; "
                    f"font-family:Avenir ;font-weight:normal;'>Hello, {name}!</h1> "
                    , unsafe_allow_html=True)
        selected = option_menu(None, ['home', 'data upload', 'annotate behaviors'],
                               icons=['house', 'file-earmark-arrow-up', 'pencil-square'],
                               menu_icon="cast", default_index=0, orientation="vertical",
                               styles={
                                   "container": {"padding": "0!important", "background-color": "#fafafa"},
                                   "icon": {"color": "black", "font-size": "18px"},
                                   "nav-link": {"color": "black", "font-size": "16px", "text-align": "center",
                                                "margin": "0px",
                                                "--hover-color": "#eee"},
                                   "nav-link-selected": {"font-size": "18px", "font-weight": "normal",
                                                         "color": "black", "background-color": "#FF6A95"},
                               }
                               )
        _, midcol, _ = st.columns([0.5, 1, 0.5])
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








