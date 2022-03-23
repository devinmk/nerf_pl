from .blender import BlenderDataset
from .llff import LLFFDataset
from .speed import PyTorchSatellitePoseEstimationDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'speed': PyTorchSatellitePoseEstimationDataset}