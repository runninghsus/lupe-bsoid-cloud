import streamlit as st


def load_view():
    st.markdown(f" <h1 style='text-align: left; color: #000000; font-size:30px; "
                f"font-family:Avenir; font-weight:normal'>Welcome to LUPE B-SOiD</h1> "
                , unsafe_allow_html=True)
    st.write("")
    desc_box = st.expander('Description', expanded=True)
    desc_box.write("LUPE B-SOiD is an automated analysis pipeline.")
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"LUPE X B-SOiD is developed by Alexander Hsu and Justin James</h1> "
                    , unsafe_allow_html=True)