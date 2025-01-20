## GCN_Spillover

### 1. Introduction
This study investigates the feasibility of enhancing financial return predictions without incorporating additional external datasets by extracting latent information within the data itself. The analysis focuses on measuring spillover effects between global financial indices and explores advanced graph embedding methodologies such as Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs)

### 2. Methods
A baseline regression model using tabular data from 15 top global indices (based on market capitalization) serves as a benchmark, where indices are used directly to predict one another. The predictive performance of these models is evaluated through repeated experiments (30 iterations) to calculate average RMSE and RMAE.

The 5 benchmark models are the following below
- Random Forest
- Gradient Boost
- Multi-Layer Perceptron
- K-Nearest Neighbors
- Support Vector Machine

Each models are tested for each of the 41 test periods from 2004 ~ 2024
[Test n : (Train, Test)]
- Test 1 : (2004.01 ~ 2004.06, 2004.07 ~ 2004.12)
- Test 2 : (2004.07 ~ 2005.12, 2005.01 ~ 2005.06)
  
  ...
  
  ...
  
  ...
- Test 41 : (2024.01 ~ 2024.06, 2024.07 ~ 2024.12)

### 3. Graph Data
Each of the test periods are converted into a graph data for future embedding modeling.

[Sample Image]
![graph_test10](https://github.com/user-attachments/assets/26ef10cf-52d2-417b-8721-e9ce914c6d56)


### 4. Results
The mean and the standard deviation of RMSE and RMAE were calculated for each model per each test period. The one - sided t-test was performed to observe if the grapb models (GCN, GAT) perform better than the benchmark ML models. The test was performed on three significant levels : 0.1, 0.05, 0.01

![image](https://github.com/user-attachments/assets/ea5c052c-6f60-41a0-b82c-65454c1d06bf)


### 5. Env & Tools
- VScode
- Conda Env
- Python 3.11.7

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from scipy.stats import ttest_ind
import itertools
from tqdm import tqdm
```






