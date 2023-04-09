import streamlit as st


def load_view():
    st.markdown(f" <h1 style='text-align: left; color: #000000; font-size:30px; "
                f"font-family:Avenir; font-weight:normal'>LUPE X B-SOiD</h1> "
                , unsafe_allow_html=True)
    st.write("")
    desc_box = st.expander('Description', expanded=True)
    desc_box.write("LUPE X B-SOiD is an automated analysis pipeline.")

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown(f'<span style="color:grey; font-family:Avenir;">'
                    f'LUPE X B-SOiD is developed by Alexander Hsu and '
                    f' Justin James </span>',
                    unsafe_allow_html=True)