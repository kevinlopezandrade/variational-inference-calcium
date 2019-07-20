import torch
import shapely
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as Fvision

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry.polygon import LinearRing


def get_non_prunned_indexes(data, coords):
    polygon = Polygon(coords) 

    N = data.shape[0]
    
    outside = []
    
    for i in range(N):
        y, x = data[i]
        
        point = Point(-x,y)
        
        if not polygon.contains(point):
            outside.append(i)
    
    
    return outside

class ConditionalDatasetPrunned(data.Dataset):

    def __init__(self, root_n, root_x, in_cluster=False):

        self.root_x = root_x
        self.root_n = root_n

        if in_cluster:
            data_n = np.load(root_n)
            data_n = torch.from_numpy(data_n)

            # Normalize to [-1, 1]
            data_n = (2 * (data_n + 815.0)/(1111.0 + 815.0)) - 1


            data_x = np.load(root_x)
            data_x = torch.from_numpy(data_x)

        else:
            data_n = np.load(root_n, mmap_mode="r")
            n = int(np.floor(data_n.shape[0] * 0.005))
            data_n = torch.from_numpy(data_n)[:n]

            # Normalize to [-1, 1]
            data_n = (2 * (data_n + 815.0)/(1111.0 + 815.0)) - 1


            data_x = np.load(root_x, mmap_mode="r")
            data_x = torch.from_numpy(data_x)[:n]


        coords = ((-3.0, -2), (-3.0, 0.0), (-1.0, 0.0), (-1.0, -2))
        non_prunned_indexes = get_non_prunned_indexes(data_x, coords)

        self.data_x = data_x[non_prunned_indexes]
        self.data_n = data_n[non_prunned_indexes]

    def __len__(self):
        return self.data_n.shape[0]

    def __getitem__(self, index):
        x = self.data_x[index]
        n = self.data_n[index]

        return n, x



class ConditionalDataset(data.Dataset):

    def __init__(self, root_n, root_x, in_cluster=False):

        self.root_x = root_x
        self.root_n = root_n

        if in_cluster:
            data_n = np.load(root_n)
            data_n = torch.from_numpy(data_n)

            # Normalize to [-1, 1]
            data_n = (2 * (data_n + 815.0)/(1111.0 + 815.0)) - 1
            self.data_n = data_n


            data_x = np.load(root_x)
            data_x = torch.from_numpy(data_x)
            self.data_x = data_x

        else:
            data_n = np.load(root_n, mmap_mode="r")
            n = int(np.floor(data_n.shape[0] * 0.005))
            data_n = torch.from_numpy(data_n)[:n]

            # Normalize to [-1, 1]
            data_n = (2 * (data_n + 815.0)/(1111.0 + 815.0)) - 1
            self.data_n = data_n


            data_x = np.load(root_x, mmap_mode="r")
            data_x = torch.from_numpy(data_x)[:n]
            self.data_x = data_x

    def __len__(self):
        return self.data_n.shape[0]

    def __getitem__(self, index):
        x = self.data_x[index]
        n = self.data_n[index]

        return n, x
