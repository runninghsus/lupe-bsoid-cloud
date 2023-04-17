import streamlit as st


def load_view():
    st.markdown(f" <h1 style='text-align: left; color: #FF6A95; font-size:30px; "
                f"font-family:Avenir; font-weight:normal'>Welcome to LUPE B-SOiD</h1> "
                , unsafe_allow_html=True)
    st.write("---")
    st.markdown(f" <h1 style='text-align: left; color: #5C5C5C; font-size:18px; "
                f"font-family:Avenir; font-weight:normal'>"
                f"LUPE X B-SOiD is a no code website for behavior analysis that "
                f"allows users to predict and analyze behavior using 2D pose estimation. "
                f"Its customized naming system fits experimental contexts, "
                f"while detailed analysis and reactive visualization design help identify trends. "
                f"Downloadable CSV files make integration with existing workflows easy.</h1> "
                , unsafe_allow_html=True)
    # desc_box = st.expander('Description', expanded=True)
    # desc_box.write("LUPE B-SOiD is an automated analysis pipeline.")
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"LUPE B-SOiD is developed by Alexander Hsu and Justin James</h1> "
                    , unsafe_allow_html=True)