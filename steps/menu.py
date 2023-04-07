import streamlit as st


def load_view():
    st.header('LUPE X B-SOiD')
    desc_box = st.expander('Description', expanded=True)
    desc_box.write("""LUPE X B-SOiD is an automated analysis pipeline.""")
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown(f'<span style="color:grey">LUPE X B-SOiD is developed by Alexander Hsu and '
                    f' Justin James </span>',
                    unsafe_allow_html=True)