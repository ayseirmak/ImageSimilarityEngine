
import torch
import numpy as np
import cv2
import pickle 


from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from lshash import LSHash

import  models
import model4
import embedding
import config
import data

class validation():
    def __init__(self,embedding_mode,similarity_mode):
        self.SimMode = similarity_mode
        self.EmbMode = embedding_mode
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.ENCODER = model4.ConvEncoder()
        self.ENCODER.load_state_dict(torch.load(config.ENC_STATE, map_location=self.DEVICE))
        self.ENCODER.eval()
        self.ENCODER.to(self.DEVICE)
        self.DATASET = data.FolderDataset(config.DATASET_PATH, config.TRANSFORMS)
        self.DATASET_EMB, self.IMG_PATH_LIST = self.get_embedding_dataset()
        
        if(similarity_mode == "LSH"):
            k = 10 # hash size
            L = 5  # number of tables
            d = self.DATASET_EMB.shape[1] # Dimension of Feature vector
            self.lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)
            
            
    def load_image_tensor(self,image_path):
        
        """
        Definition: Load image from the path, resize, normalize and convert to the tensor
        
        """
        image_tensor = config.TRANSFORMS(Image.open(image_path))
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    
    def image_embedding(self,image_path):
        """
        Definition: Create query image embedding according to the choice of Embedding mode
                    if embedding mode == Flatten; The representation matrix obtained as the result of encoder 
                                                  is directly converted to a 1-dimensional vector.
                    if embedding mode == Max_Pool; The one-dimensional vector is obtained by taking the highest 
                                                   value in each chanel of the representation matrix.
                                                   (Like Global Max Pooling)
        """
        image_tensor = self.load_image_tensor(image_path)
        image_tensor = image_tensor.to(self.DEVICE)
        
        with torch.no_grad():
            image_embedding = self.ENCODER(image_tensor)
        print(image_embedding.shape)
        
        if(self.EmbMode == "Flatten"):
           image_embedding_numpy =  image_embedding.cpu().detach().numpy()
           flattened_embedding = image_embedding_numpy.reshape((image_embedding_numpy.shape[0], -1)) 
           return flattened_embedding
        elif (self.EmbMode == "Max_Pool"):
           Max_Pool_embedding=torch.amax(image_embedding, dim=(2, 3)).cpu().detach().numpy()
           return Max_Pool_embedding
        else:
           return "Embedding Mode not sellected"
       
    def get_embedding_dataset(self):
       
        """
        Definition: Create the embedding of all images in dataset  according to the choice of Embedding mode
                    if embedding mode == Flatten; The representation matrix obtained as the result of encoder 
                                                  is directly converted to a 1-dimensional vector.
                    if embedding mode == Max_Pool; The one-dimensional vector is obtained by taking the highest 
                                                   value in each chanel of the representation matrix.
                                                   (Like Global Max Pooling)
        """
        
        """ Get Representation Matrix of all images in dataset """
        
        path = os.getcwd()
        allfiles = os.listdir(path)
        if( config.IDX_SAVE in allfiles):
            img_index = torch.load(config.IDX_SAVE , map_location=self.DEVICE)
            emb   = torch.load(config.TOTAL_EMB, map_location=self.DEVICE)
        else:    
            Dataset_loader = torch.utils.data.DataLoader(self.DATASET, batch_size=32)
            emb,img_index = embedding.create_embedding(self.ENCODER, Dataset_loader, config.EMBEDDING_SHAPE_MODEL, self.DEVICE)
            torch.save(emb, config.TOTAL_EMB)
            torch.save(img_index, config.IDX_SAVE)
            
        if(self.EmbMode == "Max_Pool"):
            max_emb =torch.amax(emb, dim=(2, 3))
            numpy_max_embedding = max_emb.cpu().detach().numpy()
            #num_images = numpy_max_embedding.shape[0]
            
            #np.save(config.EMB_SAVE_MP, numpy_max_embedding)
            #final_embedding = np.load(config.EMB_SAVE_MP)
            return numpy_max_embedding,img_index
        
        elif (self.EmbMode == "Flatten"):
            numpy_embedding = emb.cpu().detach().numpy()
            num_images = numpy_embedding.shape[0]
            flattened_embedding = numpy_embedding.reshape((num_images, -1))
           
            #np.save(config.EMB_SAVE, flattened_embedding)
            #final_embedding = np.load(config.EMB_SAVE)
            return flattened_embedding, img_index
        else:
            return "Embedding Mode not sellected"
        
    def compute_similar_images(self,image_path, num_images, num_cluster=None):
        
        img_embedding = self.image_embedding(image_path)
        path = os.getcwd()
        allfiles = os.listdir(path)
    
        if (self.SimMode == "NN_cosine"):
            if (config.KNN_NN_COSINE in allfiles):
                knn=pickle.load(open(config.KNN_NN_COSINE, 'rb'))
            else:
                knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
                knn.fit(self.DATASET_EMB)
                knnPickle = open(config.KNN_NN_COSINE, 'wb') 
                pickle.dump(knn, knnPickle)                      
                
            _, indices = knn.kneighbors(img_embedding)
            indices_list = indices.tolist()
            return indices_list
        
        elif (self.SimMode == "KMeans"):
            if(config.KMEANS in allfiles):
                kmeans = pickle.load(open(config.KMEANS, 'rb'))
            else:    
                kmeans = KMeans(n_clusters = num_cluster, random_state=0).fit(self.DATASET_EMB) 
                kmeansPickle = open(config.KMEANS, 'wb') 
                pickle.dump(kmeans, kmeansPickle)  
            labels=kmeans.labels_
            
            if(config.KNN_KMEANS in allfiles):
                knn=pickle.load(open(config.KNN_KMEANS, 'rb'))
            else:
                knn = KNeighborsClassifier(n_neighbors=num_images,algorithm='ball_tree',n_jobs=-1)
                knn.fit(self.DATASET_EMB,np.array(labels))
                knnPickle = open(config.KNN_KMEANS, 'wb') 
                pickle.dump(knn, knnPickle)  
        
            _,res = knn.kneighbors(img_embedding,return_distance=True,n_neighbors=num_images)
            res_list =res.tolist()
            return res_list
        
        elif (self.SimMode == "LSH"):
            num_images = self.DATASET_EMB.shape[0] # size: number of images + 1 First embedding is dummy embedding
            """create embedding function start with one dummy representation matrix"""
            
            for i in range(num_images-1):    
              self.lsh.index(self.DATASET_EMB[i+1])
            
            list_r=self.lsh.query(img_embedding[0],num_results=num_images,distance_func="euclidean")
            print("LSH Result:")
            print(list_r)
            return list_r
    
        else:
            return "Similarity mode not sellected"
            
    
    def return_similar_images(self,indices_list,path):
        if(self.SimMode == "LSH"):
            path_list=[]
            for j in range(config.NUM_IMG):  
                for key in indices_list[j]:
                    print(type(np.asarray(key)))
                    index=np.where((self.DATASET_EMB == np.asarray(key)).all(axis=1))
                    print(index)
                    for i in index[0]:
                        if self.IMG_PATH_LIST[i-1] not in path_list:
                            file_path = os.path.join(config.DATASET_PATH,self.IMG_PATH_LIST[i-1])
                            path_list.append(file_path)
                            # image = cv2.imread(file_path)
                            # plt.imshow(image)
                            # plt.show()
                   # fig = plt.figure(figsize=(8, 8))
                   # for i in range(1,len(path_list)+1):
                   #    image = cv2.imread(file_path)
                   #    fig.add_subplot(2,4,i)
                   #    plt.imshow(image)
                   # plt.show()
            print(path_list)
            return path_list
        
        else:    
            img_list=[]
            indices = indices_list[0]
            for index in indices:
                if index == 0:
                    # index 0 is a dummy embedding.
                    pass
                else:
                    transforms=T.ToPILImage();
                    a=self.DATASET[int(index)-1][0]
                    img = transforms(a)
                    img_list.append(img)
                    print(self.IMG_PATH_LIST[int(index)-1])
                    # plt.imshow(img)
                    # plt.show()
            fig = plt.figure(figsize=(8, 8))
            for i in range(1,len(img_list)+1):
                fig.add_subplot(2,4,i)
                plt.imshow(img_list[i-1])
            #plt.savefig(path)
            plt.show()
          
            return img_list  
         
    def load_autoencoder_state(self):
    
        # Load  state  of encoder
        self.ENCODER.load_state_dict(torch.load(config.ENC_STATE, map_location=self.DEVICE))
        self.ENCODER.eval()
        self.ENCODER.to(self.DEVICE)
        
        #Load state of decoder
        # self.DECODER.load_state_dict(torch.load(config.DEC_STATE, map_location=self.DEVICE))
        # self.DECODER.eval()
        # self.DECODER.to(self.DEVICE)
    
    
    
    def result(self):
        indices_list = self.compute_similar_images(config.IMAGE_PATH,config.NUM_IMG, 50)      
        return indices_list     
               
    
        
if __name__ == "__main__":                                    
    val = validation("Max_Pool","KMeans")    
    #val.load_autoencoder_state() 
    img_list = val.result()
    val.return_similar_images(img_list,"Result_dataset2/Model4/Flatten-NNcosine")
    