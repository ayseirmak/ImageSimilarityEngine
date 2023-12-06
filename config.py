#Configuration
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

DATASET_PATH="dataset_v1"
MAX_LOSS=0.05
BATCH_SIZE=16
LR=0.001
EPOCHS=40
TRAIN_RATIO=0.85

IMAGE_PATH="dataset_v1/0.jpg"

EMB_SAVE_MP="data_embedding_MP.npy"
EMB_SAVE_F="data_embedding_F.npy"
EMB_SAVE_F="data_embedding_F.npy"

PATH="Result_dataset1/"
PATH_EMB=PATH+"Model6/ReduceLR/"
TOTAL_EMB="total_emb.pt"
IDX_SAVE="emb_path_list.pt"

KNN_NN_COSINE = "KNN_NNcosine.pkl"
KNN_KMEANS = "KNN_KMeans.pkl"
KMEANS = "KMEANS.pkl"

ENC_STATE=PATH+"Model6/ReduceLR/ENCODER_model6_lr.pt"
DEC_STATE=PATH+"Model6/ReduceLR/DECODER_model6_lr.pt"
NUM_IMG=8

EMBEDDING_SHAPE_MODEL = (1, 1024, 2, 2) # This we know from our encoder - model1
#EMBEDDING_SHAPE_MODEL2 = (1, 256, 16, 16) # This we know from our encoder

TRANSFORMS = T.Compose([T.Resize((224,224)),T.ToTensor()])