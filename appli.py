# if __name__ == '__appli__':
import streamlit as st
from sections.nailsdetection.nails import side_bar_nails, initalize_variables, print_info, treatment_nails
from sections.regression.regression import regression_page
from sections.classification.classification_tab_contents import tab2_content, tab3_content,tab4_content, tab5_content
from sections.classification.Preparation_df import classification_page
from sections.Home.Home_page_print import print_images
from sections.nailsdetection.nails_vid import nails_page

st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

type_data = st.sidebar.radio(
    "Choose your playground",
    ["Home","Regression", "Classification", "NailsDetection"]
)
if type_data == "Home":
    st.write('# Home page ')
    print_images()

elif type_data == "Regression":
    regression_page()
 
elif type_data == "Classification":
    tab_visit = "None"
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preparation", "Analysis", "Play with parameters","Try a CNN","Use your models"])
    with tab1:

        df_prep = classification_page()
    with tab2:
        tab_visit, liste_col, list_choice2, df = tab2_content(tab_visit,df_prep)
    with tab3:
        tab3_content(tab_visit, liste_col, list_choice2, df)
    with tab4:
        st.header("Try a CNN")
        tab4_content(tab_visit,list_choice2, df)
    with tab5:
        st.header("Use the models you have trained")
        tab5_content()

elif type_data == "NailsDetection":
    tab1, tab2 = st.tabs(
        ["Image", "Video"])
    with tab1:
        st.write('# Detection of nails from an image')
        uploaded_file,confidence_threshold = side_bar_nails()
        prediction, polygonPoints, nails = initalize_variables()
        print_info()
        treatment_nails(uploaded_file, confidence_threshold,prediction,polygonPoints)
    with tab2:
        # Main section: Title of the app and a subheading indicating the output will show an inferenced video
        st.write('# Detection of nails from a stream')
        st.write('### Inferenced Video')
        gif_url = "https://beautyinsider.sg/wp-content/uploads/2020/02/Thumbnail-Rihanna-Nails-Manicure-2.gif"
        st.markdown(f"<img src='{gif_url}' width='300'>", unsafe_allow_html=True)
        # Appel de la page de détection d'ongles
        nails_page()


else:
    st.write("Choose an option")
