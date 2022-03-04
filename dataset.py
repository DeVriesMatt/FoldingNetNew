import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from pyntcloud import PyntCloud


class PointCloudDataset(Dataset):
    def __init__(self, annotations_file,
                 img_dir,
                 img_size=400,
                 label_col='Treatment',
                 transform=None,
                 target_transform=None,
                 centring_only=False):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.centring_only = centring_only

        self.new_df = self.annot_df[(self.annot_df.xDim <= self.img_size) &
                                    (self.annot_df.yDim <= self.img_size) &
                                    (self.annot_df.zDim <= self.img_size)].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df['label_col_enc'] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, 'Treatment']
        img_path = os.path.join(self.img_dir, treatment, self.new_df.loc[idx, 'serialNumber'])
        image = PyntCloud.from_file(img_path + '.ply')
        image = torch.tensor(image.points.values)
        # TODO: take away after testing
        if self.centring_only:
            mean = torch.mean(image, 0)
        # mean = torch.tensor([[13.4828, 26.5144, 24.4187]])
        # std = torch.tensor([[9.2821, 20.4512, 18.9049]])
            image = (image - mean)
        # / std
        # TODO: _____________________________________________
        else:
            mean = torch.tensor([[13.4828, 26.5144, 24.4187]])
            std = torch.tensor([[9.2821, 20.4512, 18.9049]])
            image = (image - mean) / std
        # return encoded label as tensor
        label = self.new_df.loc[idx, 'label_col_enc']
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        return image, label, feats


class PointCloudDatasetAll(Dataset):
    def __init__(self, annotations_file,
                 img_dir,
                 img_size=400,
                 label_col='Treatment',
                 transform=None,
                 target_transform=None,
                 centring_only=False,
                 cell_component='cell'):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.centring_only = centring_only
        self.cell_component = cell_component

        self.new_df = self.annot_df[(self.annot_df.xDim <= self.img_size) &
                                    (self.annot_df.yDim <= self.img_size) &
                                    (self.annot_df.zDim <= self.img_size)].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df['label_col_enc'] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, 'Treatment']
        plate_num = 'Plate' + str(self.new_df.loc[idx, 'PlateNumber'])
        if self.cell_component == 'cell':
            component_path = 'stacked_pointcloud'
        else:
            component_path = 'stacked_pointcloud_nucleus'

        img_path = os.path.join(self.img_dir,
                                plate_num,
                                component_path,
                                treatment,
                                self.new_df.loc[idx, 'serialNumber'])
        image = PyntCloud.from_file(img_path + '.ply')
        image = torch.tensor(image.points.values)

        # TODO: take away after testing
        if self.centring_only:
            mean = torch.mean(image, 0)
            image = (image - mean)

        else:
            mean = torch.tensor([[13.4828, 26.5144, 24.4187]])
            std = torch.tensor([[9.2821, 20.4512, 18.9049]])
            image = (image - mean) / std

        # TODO: _____________________________________________
        # return encoded label as tensor
        label = self.new_df.loc[idx, 'label_col_enc']
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        return image, label, feats
