import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
import os 
import cv2 
import torch


model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
model.classifier.fc = torch.nn.Identity()



# model = EfficientNet.from_pretrained('efficientnet-b0')
# # by pass the fc layer and get the embeddings as output of model 
# model._fc = torch.nn.Identity()


embeddings=[]
files = []
# df = pd.DataFrame(columns=['image_id', 'embedding'])

folder_path = '16k_images'
for file in os.listdir(folder_path):
    # reading the image
    img = plt.imread(os.path.join(folder_path, file))

    # normalizing the image to -1 to 1 range 
    img = img/127.5 - 1

    # resize the image to make it compatible with the model
    img = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)

    # convert to tensor
    img = torch.from_numpy(img).float()

    # store the embeddings and image name in a list and dataframe (if gray scale image skip the image)
    try:
        embedding = model(img.reshape(1, 3, 224, 224)).detach().numpy()
        # df = df.append({'image_id': file, 'embedding': embedding},ignore_index=True)
        embeddings.append(embedding)
        files.append(file)
    except:
        print("error at:",file)
        
np.save('image_names.npy',files)
np.save('embeddings.npy',embeddings)
