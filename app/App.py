import streamlit as st
import pandas as pd
import time
import os
from PIL import Image
import shutil

import config
from validation2 import validation
from pyunpack import Archive

from streamlit_option_menu import option_menu

import text
QI_PATH = "App_QI"
DS_PATH = "App_DS"
st.set_page_config(page_title="Image Similarity Engine", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 400px;
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)
@st.cache
def load_image( image_file ):
    img = Image.open( image_file )
    return img

def save_dir(img_path,uploadfile):
    with open(img_path,"wb") as f:
         f.write(uploadfile.getbuffer())
         
def save_uploadedfile():
   with colum1:
    if st.session_state.QI_upload is not None:
        img_path = os.path.join(QI_PATH,st.session_state.QI_upload.name)
        save_dir(img_path,st.session_state.QI_upload)
        st.image(load_image(img_path))
        config.IMAGE_PATH = img_path
        return st.success("Query IMAGE: " +st.session_state.QI_upload.name +" successfully saved")    


def save_zipfile():

    if st.session_state.dataset_upload is not None:
        Dataset_Details = {"FileName":st.session_state.dataset_upload.name,"FileType":st.session_state.dataset_upload.type}
        st.write( Dataset_Details)
        zip_path = os.path.join(DS_PATH,st.session_state.dataset_upload.name)
        save_dir(zip_path,st.session_state.dataset_upload)
        Archive(zip_path).extractall(DS_PATH)
        os.remove(zip_path)
        if(len(os.listdir(DS_PATH))==1):
            for i in os.listdir(DS_PATH):
                ds_path = os.path.join(DS_PATH,i)
                if os.path.isdir(ds_path):
                    #st.write("dfdssdg")
                    config.DATASET_PATH = ds_path
                    #st.write(config.DATASET_PATH)
            return st.success("Dataset: " +st.session_state.dataset_upload.name +" successfully saved") 
        else: st.write("path length is not 1")
                        

def set_num_similar_img ():
    config.NUM_IMG=st.session_state.num_img

                
def validation_process(): 
    MP=[]
    Flt=[]
    for i in st.session_state["Embedding_type"]:
        if (i == "Max Pool"):
            if(st.session_state["cosine_MP"]):
                MP.append("cosine")
            if(st.session_state["euc_MP"]):
                MP.append("euclidean")
            
        elif(i == "Flatten"):
            if(st.session_state["cosine_F"]):
                Flt.append("cosine")
            if(st.session_state["euc_F"]):
                Flt.append("euclidean")
    container.subheader("Results")
    if(MP):
        for i in MP:
            val=validation("Max_Pool",i)
            img_list = val.result()
            image_path_list,img_name=val.return_similar_images(img_list,"")
            container.write("**Embedding Method:  _Max Pool_**")
            container.write("**Distance Metric: "+"_"+i+"_**")
            container.image(image_path_list,img_name)
    if(Flt):
        for i in Flt:
            val=validation("Flatten",i)
            img_list = val.result()
            image_path_list,img_name=val.return_similar_images(img_list,"")
            container.write("**Embedding Method:  _Flatten_**")
            container.write("**Distance Metric: "+"_"+i+"_**")
            container.image(image_path_list,img_name)



choose = option_menu("Image Similarity Engine", ["Home", "Engine","Datasets", "Model", "Training", "Contact"],
                          icons=['house', "layers-half",'clipboard-data', 'boxes', 'columns-gap','person lines fill'],
                          menu_icon="app-indicator", default_index=0,
                          orientation="horizontal",
                          key="page",
                          styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "30px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"}
                     
    }
    )


if(choose=="Home"): 
    st.title('Welcome to the Image Similarity Engine')
    st.markdown(text.DEFINITION)
    st.subheader("Purpose: ")
    st.write(text.PURPOSE)
    st.subheader("Stages:")
    st.write("**_For Quick Start Go to sidebar and follow the below steps:_**")
    st.write(text.STAGES)
    st.subheader("Used Technologies and Strategy: ")
    st.write(text.UT)
    st.subheader("Factors To Consider:")
    st.write(text.FTC)
    container = st.container()
    
elif(choose=="Engine"):
    st.title("Engine Structer")
    st.subheader("Example Submissions and Results:")
    st.write("*Dataset 1 - Animal Faces*")
    c1,c2=st.columns(2)
    c1. write("**Query Image**")
    c1.image("dataset_v2/Alfred_Sisley_240.jpg",width=200)
    c1. write("**Result**")
    c1.image("deneme.png")
    st.subheader("Flowchart")
    colum1,colum2=st.columns([8,3])
    colum1.image("AS2.png")
elif(choose=="Datasets"):
    st.title("Pretrained Datasets")
    st.subheader("Structer of Datasets")
    data = {
  "Name": ["Animal Faces", "Best Artworks of All Time "],
  "Number of Image": [4738, 8355],
  "Description": ["Contains six types of high quality wild animal face", "Includes nearly 200 color and pencil drawings by fifty different artists."]
      }
    df=pd.DataFrame(data)
  
    colum1,colum2=st.columns(2)
    colum1.write("**Animal Faces**")
    colum1.image("d1.png")
    colum1.write("**Best Artworks of All Time**")
    colum1.image("d2.png")
    colum2.write("**Description**")
    colum2.table(df)
    
   

with st.sidebar:
    st.header("QUICK START")
    st.subheader("Stage 1")
    st.write("**Upload your Selected image from your computer**" )
    colum1,colum2= st.sidebar.columns([10,1]) 
    with colum1:
        QI_uploaded_file = st.file_uploader("Upload Query Image",type=['png','jpeg','jpg'],key="QI_upload",on_change = save_uploadedfile)            
    
    st.subheader("Stage 2")
    if "dataset_option" not in st.session_state:
         st.session_state["dataset_option"]="zzz"
    st.write("**Select whether you prefer pre-trained datasets or the dataset you will upload**")
    dataset_option = st.selectbox("",("Select your dataset preference","Upload own dataset", 'Search in pretrained datasets'),index=0,key="dataset option")
  
    if(dataset_option == "Upload own dataset"):
          st.write("**Select the compressed file of the dataset you want to upload.**",)    
          dataset_uploaded_file = st.file_uploader("Uploaded file shoulde be 7-Zip, Zip or RAR compressed file.",
                                                   type=['zip','tar','rar'], key = "dataset_upload", 
                                                   on_change=save_zipfile)
    elif(dataset_option == "Search in pretrained datasets") :
        st.write("**Select one of the specified datasets**")
        dataset = st.radio("You can look at the general structure of datasets from the homepage.",
                          ('Animal Faces', 'Best Artworks of All Time'),key="dataset")

    st.subheader("Stage 3")
    st.write("**Insert how many images you want to find most similar to Query Image from the uploaded dataset.**") 
    num_img = st.number_input("",min_value=1,value=6,step=1,help="Insert the number of similar images",key="num_img",on_change = set_num_similar_img)    
    
    st.subheader("Stage 4")
    st.write("**Choose Embedding Methods and Distance Metrics**")
    Embedding_selectbox = st.multiselect(label="",
                                        options=(["Max Pool","Flatten"]),
                                        default = None,
                                        key = "Embedding_type")
                                        
    c1,c2= st.sidebar.columns(2)  
    for i in st.session_state["Embedding_type"]:
        if (i == "Max Pool"):
            with c1:
                st.write("**Embedding Method: Max Pool**")
                st.write("Check the distance metric(s) to be used with this embedding mode")
                Cosine_MP = st.checkbox('Cosine Distance',  key = "cosine_MP")
                Euc_MP = st.checkbox('Euclidean Distance',key = "euc_MP")
        elif(i == "Flatten"):
            with c2:
                st.write("**Embedding Method: Flatten**")
                st.write("Check the distance metric(s) to be used with this embedding mode")
                cosine_F = st.checkbox('Cosine Distance',key = "cosine_F")
                euc_F = st.checkbox('Euclidean Distance',key = "euc_F")

    submit_button = st.button("Submit")
    st.write("")
    if submit_button:
        config.NUM_IMG=st.session_state.num_img
        if (st.session_state["dataset"]=="Animal Faces"):
            config.DATASET_PATH="dataset_v1/"
            config.PATH="Result_dataset1/"
        else:
            config.DATASET_PATH="dataset_v2/"
            config.PATH="Result_dataset2/" 
        validation_process()
        choose="Engine"


    
    # embedding_mode,similarity_mode,metric
    # val = validation()    
    # #st.write("dfdff")
    # val.load_autoencoder_state() 
    # #st.write("dfdff")
    # index_list = val.result()
    # #st.write("dfdff")
    # image_list=val.return_similar_images(index_list)
    # #st.write(a)
    # container.image(image_list)        
   
# from PIL import Image
# image = Image.open('dataset_v1/0.jpg.')
# a=[]
# a.append(image)
# a.append(image)
# st.image(a,use_column_width="always")

# # #Spinner waiting process
# # with st.spinner('Wait for it...'):
# #     time.sleep(5)
# # st.success('Done!')