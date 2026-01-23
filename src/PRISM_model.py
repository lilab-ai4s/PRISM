#M1
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from typing import Optional

import scanpy as sc
import anndata as ad
import diopy
from scipy import sparse
from scipy.sparse import issparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed=42


#M1
class M1(nn.Module):
    """
    logits = fc_shared + α·fc6 + γ·fc_marker
    """
    def __init__(self, input_size, h1, h2, h3, h4,
                 num_classes,
                 marker_mask,      
                 freeze_marker=True,
                 alpha=2.0,         
                 gamma=5):        
        super().__init__()


        self.fc1 = nn.Linear(input_size, h1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(h2, h3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(h3, h4)
        self.relu4 = nn.ReLU()
        self.fc_shared = nn.Linear(h4, num_classes)     
        self.fc_skip = nn.Linear(input_size, num_classes)


        self.fc_marker = nn.Linear(input_size, num_classes, bias=False)
        with torch.no_grad():

           
            self.fc_marker.weight.copy_(marker_mask)
            self.fc_marker.weight.div_(marker_mask.sum(1, keepdim=True).clamp(min=1))
        if freeze_marker:
            for p in self.fc_marker.parameters():
                p.requires_grad = False   

        self.alpha = alpha       
        self.gamma = gamma       

    def forward(self, x):

        h = self.relu1(self.fc1(x))
        h = self.relu2(self.fc2(h))
        h = self.relu3(self.fc3(h))
        h = self.relu4(self.fc4(h))
        logits = self.fc_shared(h)

        logits = logits + self.alpha * self.fc_skip(x)      
        logits = logits + self.gamma * self.fc_marker(x)    
        return logits
    

class M2(nn.Module):
    """
    logits = fc_shared + α·fc_skip
        
    """
    def __init__(self, input_size, h1, h2, h3, h4,
                 num_classes,
                 marker_mask,       
                 anti_mask,         
                 freeze_marker=True,
                 freeze_anti=True,
                 alpha=1.0,         
                 gamma=5.0,         
                 delta=5.0):        
        super().__init__()


        self.fc1 = nn.Linear(input_size, h1); self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h1, h2);         self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(h2, h3);         self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(h3, h4);         self.relu4 = nn.ReLU()
        self.fc_shared = nn.Linear(h4, num_classes)
        self.fc_skip = nn.Linear(input_size, num_classes)

        self.fc_marker = nn.Linear(input_size, num_classes, bias=False)
        with torch.no_grad():
            w = marker_mask / marker_mask.sum(1, keepdim=True).clamp(min=1)
            self.fc_marker.weight.copy_(w)
        if freeze_marker:
            for p in self.fc_marker.parameters(): p.requires_grad = False


        self.fc_anti = nn.Linear(input_size, num_classes, bias=False)
        with torch.no_grad():
            w = anti_mask / anti_mask.sum(1, keepdim=True).clamp(min=1)
            self.fc_anti.weight.copy_(-w)               
        if freeze_anti:
            for p in self.fc_anti.parameters(): p.requires_grad = False

        self.alpha, self.gamma, self.delta = alpha, gamma, delta

    def forward(self, x):
  
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.relu3(self.fc3(out))
        out = self.relu4(self.fc4(out))
        logits = self.fc_shared(out)         

        
        logits += self.alpha * self.fc_skip(x)   
        logits += self.gamma * self.fc_marker(x)  
        logits += self.delta * self.fc_anti(x)    

        return logits


def fmap_train_retrain_st1(sc_data, st_data, result_save_path, num_classes, anno, marker_mask, num_epochs=100):

    type_num=len(set(sc_data.obs[anno]))
    num_classes = len(set(sc_data.obs[anno]))
    cell_types = sc_data.obs[anno]
    cell_types_categorical = cell_types.astype("category")
    cell_types_integer = cell_types_categorical.cat.codes
    sc_data.obs["type_integer"] = cell_types_integer
    
    num_list = list(range(1, len(st_data.obs)))
    num_list = list(range(1, len(sc_data.obs)))
    split_idx = int(len(num_list) * 0.8)

    train_data = sc_data[:split_idx, :]
    valid_data = sc_data[split_idx:, :]
    X_train = train_data.X.todense()
    X_valid = valid_data.X.todense()


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)

    y_train = train_data.obs["type_integer"].values
    y_valid = valid_data.obs["type_integer"].values

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)



    

    for i in range(10):
       
        seed = i
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        input_size = sc_data.X.shape[1]
        hidden_size1 = 256
        hidden_size2 = 128
        hidden_size3 = 64
        hidden_size4 = 32

        learning_rate = 0.01
        num_epochs = num_epochs

        model = M1(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes, marker_mask).to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
       
        criterion = nn.CrossEntropyLoss()
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        X_valid_tensor = X_valid_tensor.to(device)
        y_valid_tensor = y_valid_tensor.to(device)


        for epoch in range(num_epochs):
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    outputs = model(X_valid_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_valid_tensor.size(0)
                    correct += (predicted == y_valid_tensor).sum().item()

                    

                model.train()
                
                
    



        new_data_X = st_data.X.todense()
        new_data_tensor = torch.tensor(new_data_X, dtype=torch.float32).to(device)  

        model = model.to(device)  
        model.eval()



        with torch.no_grad():
            outputs = model(new_data_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)



        probabilities_np = probabilities.cpu().numpy()
        probabilities_df = pd.DataFrame(probabilities_np, columns=cell_types_categorical.cat.categories)
        result_save_path2=result_save_path+str(i)+".csv"
        probabilities_df.to_csv(result_save_path2, index=False)
        
        


def fmap_train_retrain_st2(sc_data, st_data, result_save_path, model_save_path, plot_save_path, num_classes, anno, marker_mask, anti_mask, num_epochs=200):    
    special_colors=None
   
    type_num=len(set(sc_data.obs[anno]))
    num_classes = len(set(sc_data.obs[anno]))
    cell_types = sc_data.obs[anno]
    cell_types_categorical = cell_types.astype("category")
    cell_types_integer = cell_types_categorical.cat.codes
    sc_data.obs["type_integer"] = cell_types_integer
    
    num_list = list(range(1, len(st_data.obs)))
    num_list = list(range(1, len(sc_data.obs)))
    split_idx = int(len(num_list) * 0.8)

    train_data = sc_data[:split_idx, :]
    valid_data = sc_data[split_idx:, :]
    X_train = train_data.X.todense()
    X_valid = valid_data.X.todense()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)

    y_train = train_data.obs["type_integer"].values
    y_valid = valid_data.obs["type_integer"].values

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)




    for i in range(1):
       
        seed = i
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        input_size = sc_data.X.shape[1]
        hidden_size1 = 256
        hidden_size2 = 128
        hidden_size3 = 64
        hidden_size4 = 32
        learning_rate = 0.01
        num_epochs = num_epochs
        model = M2(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes, marker_mask,anti_mask).to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        X_valid_tensor = X_valid_tensor.to(device)
        y_valid_tensor = y_valid_tensor.to(device)

        for epoch in range(num_epochs):
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    outputs = model(X_valid_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_valid_tensor.size(0)
                    correct += (predicted == y_valid_tensor).sum().item()

                    

                model.train()
                
                
        model_save_path2=model_save_path+str(i)+".pth"
        torch.save(model.state_dict(), model_save_path2)



        new_data_X = st_data.X.todense()
        new_data_tensor = torch.tensor(new_data_X, dtype=torch.float32).to(device)  

        model = model.to(device)  
        model.eval()



        with torch.no_grad():
            outputs = model(new_data_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)


        probabilities_np = probabilities.cpu().numpy()
        probabilities_df = pd.DataFrame(probabilities_np, columns=cell_types_categorical.cat.categories)
        
        
        result_save_path2=result_save_path+str(i)+".csv"
        probabilities_df.to_csv(result_save_path2, index=False)
        predicted_labels = cell_types_categorical.cat.categories[predicted.cpu()]
        st_data.obs['predicted_classes'] = predicted_labels
        predicted_labels = st_data.obs['predicted_classes']
        

        st_obj2=sc_data

        unique_clusters = np.unique(st_obj2.obs[anno])
        unique_predictions = np.unique(st_data.obs['predicted_classes'])

        cmap_clusters = plt.get_cmap('tab20', len(unique_clusters))
        cmap_predictions = plt.get_cmap('tab20b', len(unique_predictions))  

        color_map = {label: cmap_clusters(i) for i, label in enumerate(unique_clusters)}
        color_map.update({label: cmap_predictions(i) for i, label in enumerate(unique_predictions) if label not in color_map})

        
   

        colors = [color_map[label] for label in st_data.obs['predicted_classes']]
        from matplotlib.colors import to_rgba
        
        colors = [to_rgba(color_map[label]) for label in st_data.obs['predicted_classes']]
        subset_color_map = {label: color_map[label] for label in unique_predictions if label in color_map}
        
   
        coor_x = st_data.obs['x']
        coor_y = st_data.obs['y']

        # plt.figure(figsize=(15, 15))
        fig, ax = plt.subplots(figsize=(15, 15))
        scatter = ax.scatter(coor_x, coor_y, c=colors, s=10)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=cell_type, 
                                  markerfacecolor=color, markersize=15) 
                           for cell_type, color in subset_color_map.items()]
        
        ax.legend(handles=legend_elements, title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')
        result=pd.read_csv(result_save_path2)
     
        plot_save_path2=plot_save_path+str(seed)+".pdf"

        plt.savefig(plot_save_path2,format='pdf', bbox_inches='tight', dpi=300)
        
        plt.close()
