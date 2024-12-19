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

    -  [x] [📄DocumentGPT](http://localhost:8501/DocumentGPT)
    -  [ ] [🔒PrivateGPT](http://localhost:8501/PrivateGPT)
    -  [x] [❓QuizGPT](http://localhost:8501/QuizGPT)
    -  [x] [🖥️SiteGPT](http://localhost:8501/SiteGPT)
    -  [x] [🕓MeetingGPT](http://localhost:8501/MeetingGPT)
    -  [ ] [💰InvestorGPT](http://localhost:8501/InvestorGPT)
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
