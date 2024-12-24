import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)

st.title("FullstackGPT Home")

st.markdown(
    """
    ### Hello!

    Welcome to my FullstackGPT Portfolio!
    Here are the apps I made:

    -  [x] [📄DocumentGPT](http://smin987.streamlit.app/DocumentGPT)
    -  [ ] [🔒PrivateGPT]
    -  [x] [❓QuizGPT](http://smin987.streamlit.app/QuizGPT)
    -  [x] [🖥️SiteGPT](http://smin987.streamlit.app/SiteGPT)
    -  [ ] [🕓MeetingGPT]
    -  [ ] [💰InvestorGPT]
    """
)
# with st.sidebar:
#     st.title("sidebar title")
#     st.text_input("xxx")

# tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])

# with tab_one:
#     st.write("a")

# with tab_two:
#     st.write("b")

# with tab_three:
#     st.write("c")
#     st.write("c")
