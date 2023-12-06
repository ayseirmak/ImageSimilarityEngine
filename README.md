## Image Similarity Engine via AutoEncoder
An image similarity search engine has been developed which identifies images similar to the image selected by the user from a dataset. The latent vectors of the images are generated by a CNN based autoencoder model. The nearest neighbour algorithm was used to find images similar to the selected image. The similarity measure used by the algorithm (Euclidean/Cosine) and the methods used to obtain the latent vector (MaxPoll/Flatten) were obtained from the user. An image similarity search engine platform was created to visualise to demonstraite research. Within the scope of this research, CNN autoencoders with 6 different architectures were created. Each autoencoder was trained on 2 different data sets. In addition, different latent vector generation methods and different similarity finding methods were tested and performance was compared.

## Modules:
The study consists of two modules: Image Similarity Model and User Interface Platform.

(1) In the image similarity model, suitable compressed representation vectors of the images in the dataset specified by the user and the image selected by the user are generated. Within the scope of the similarity metric specified by the user, the closest vectors to the representation vector of the selected image are found among the extracted representation (latent) vectors by using the nearest neighbour algorithm. The images to which the found vectors belong are stored.
### Workflow
![image](https://github.com/ayseirmak/ImageSimilarityEngine/assets/152956281/9fb5a334-c87e-4cf1-92d9-1876cfa62897)

(2) The user interface module is a platform where the user can enter the parameters determined for the image similarity model. The image similarity model is executed according to the parameters given by the user, and the images obtained as a result of the model are presented to the user through this platform.
### User Interface
![image](https://github.com/ayseirmak/ImageSimilarityEngine/assets/152956281/1bca4b79-5652-4012-b0c5-b1fd8ca77935)

## Best CNN-AutoeEncoder Architecture and Performance
<img width="1298" alt="image" src="https://github.com/ayseirmak/ImageSimilarityEngine/assets/152956281/c9a78352-7de4-4536-91d0-fd517981385e">
<img width="186" alt="image" src="https://github.com/ayseirmak/ImageSimilarityEngine/assets/152956281/52e4ae61-7edc-4141-8dda-9f8ca5311ecd">
<img width="1101" alt="image" src="https://github.com/ayseirmak/ImageSimilarityEngine/assets/152956281/c56a6546-b0e3-4a15-82c9-d42a18835e89">

When analysing the experiments, it was observed that the model obtained by training model 6, which was created as part of this study, using the ReduceLROnPlateue modifier showed high performance compared to other models. In this case, it was observed that as the latent vector size decreases and the number of channels increases, better quality models emerge with appropriate training parameters. In addition, when analysing the experiments, it was observed that using the Max Pool method and choosing the Cosine distance metric resulted in higher similarity performance compared to other <Embedding Method, Distance Metric> combinations.

You can run App in your local by using datsets ([dataset1](https://www.kaggle.com/datasets/vishweshsalodkar/wild-animals), [dataset2](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)), models and streamlit App in the repo.
Please ensure that you should setup your environment as follow:

Framework|Version
------------- | -------------
python| 3.8
pytorch| 1.11
streamlit|1.10.0
pillow|9.0.1 
numpy|1.21.5 
matplotlib|3.5.1 
scikit-learn|1.1.1 
tensorboard|2.9.0 
scipy|1.8.1 
lshash3|	0.0.8 
opencv-python|4.6.0.66 


