"""
This code is adapted from https://github.com/nyuad-cai/MedFuse
"""


import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

from classification.datasets.med_lab_utils.preprocessor import Discretizer, Normalizer
from omegaconf import DictConfig
import hydra

R_CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']

CLASSES = [
       'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
       'Acute myocardial infarction', 'Cardiac dysrhythmias',
       'Chronic kidney disease',
       'Chronic obstructive pulmonary disease and bronchiectasis',
       'Complications of surgical procedures or medical care',
       'Conduction disorders', 'Congestive heart failure; nonhypertensive',
       'Coronary atherosclerosis and other heart disease',
       'Diabetes mellitus with complications',
       'Diabetes mellitus without complication',
       'Disorders of lipid metabolism', 'Essential hypertension',
       'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
       'Hypertension with complications and secondary hypertension',
       'Other liver diseases', 'Other lower respiratory disease',
       'Other upper respiratory disease',
       'Pleurisy; pneumothorax; pulmonary collapse',
       'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
       'Respiratory failure; insufficiency; arrest (adult)',
       'Septicemia (except in labor)', 'Shock'
    ]

ETHNICITY = {'WHITE': 0,
 'UNKNOWN': 1,
 'OTHER': 2,
 'BLACK/AFRICAN AMERICAN': 3,
 'HISPANIC/LATINO': 4,
 'ASIAN': 5,
 'AMERICAN INDIAN/ALASKA NATIVE': 6,
 'UNABLE TO OBTAIN': 7}

GENDER = {'M': 0, 'F': 1}

LAST_CAREUNIT = {'Medical Intensive Care Unit (MICU)': 0,
 'Cardiac Vascular Intensive Care Unit (CVICU)': 1,
 'Coronary Care Unit (CCU)': 2,
 'Surgical Intensive Care Unit (SICU)': 3,
 'Trauma SICU (TSICU)': 4,
 'Medical/Surgical Intensive Care Unit (MICU/SICU)': 5,
 'Neuro Stepdown': 6,
 'Neuro Surgical Intensive Care Unit (Neuro SICU)': 7}


class MIMIC_CXR_EHR(Dataset):
    def __init__(self, cfg, metadata_with_labels, ehr_ds, cxr_ds, split='train'):
        
        self.CLASSES = CLASSES
        if 'radiology' in cfg.dataset.labels_set:
            self.CLASSES = R_CLASSES
        
        self.metadata_with_labels = metadata_with_labels
        self.cxr_files_paired = self.metadata_with_labels.dicom_id.values
        self.ehr_files_paired = (self.metadata_with_labels['stay'].values)
        self.cxr_files_all = cxr_ds.filenames_loaded
        self.ehr_files_all = ehr_ds.names
        self.ehr_files_unpaired = list(set(self.ehr_files_all) - set(self.ehr_files_paired))
        self.ehr_ds = ehr_ds
        self.cxr_ds = cxr_ds
        self.cfg = cfg
        self.split = split
        self.data_ratio = self.cfg.dataset.data_ratio 
        if split=='test':
            self.data_ratio =  1.0
        elif split == 'val':
            self.data_ratio =  0.0


    def __getitem__(self, index):
        meta_info = {}
        age = None
        gender = None
        ethnicity = None
        if self.cfg.dataset.data_pairs == 'paired_ehr_cxr':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
            meta_info['id_ehr'] = self.ehr_files_paired[index]
            meta_info['id_cxr'] = self.cxr_files_paired[index]
            
            age = self.metadata_with_labels.iloc[index]['age'] / 91.0 # Age is capped to 91 in MIMIC
            # Gotta One Hot Encode this.
            gender = [1 if key == self.metadata_with_labels.iloc[index]['gender'] else 0 for key in GENDER.keys()]
            ethnicity = [1 if key == self.metadata_with_labels.iloc[index]['ethnicity'] else 0 for key in ETHNICITY.keys()]
            return ehr_data, cxr_data, labels_ehr, labels_cxr, meta_info, age, gender, ethnicity
        elif self.cfg.dataset.data_pairs == 'paired_ehr':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr, meta_info, age, gender, ethnicity
        elif self.cfg.dataset.data_pairs == 'radiology':
            ehr_data, labels_ehr = np.zeros((1, 10)), np.zeros(self.args.num_classes)
            cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_all[index]]
            return ehr_data, cxr_data, labels_ehr, labels_cxr, meta_info, age, gender, ethnicity
        elif self.cfg.dataset.data_pairs == 'partial_ehr':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_all[index]]
            cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr, meta_info, age, gender, ethnicity
        
        elif self.cfg.dataset.data_pairs == 'partial_ehr_cxr':
            if index < len(self.ehr_files_paired):
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
                cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
            else:
                index = random.randint(0, len(self.ehr_files_unpaired)-1) 
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_unpaired[index]]
                cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr, meta_info, age, gender, ethnicity, last_careunit

        
    
    def __len__(self):
        if 'paired' in self.cfg.dataset.data_pairs:
            return len(self.ehr_files_paired)
        elif self.cfg.dataset.data_pairs == 'partial_ehr':
            return len(self.ehr_files_all)
        elif self.cfg.dataset.data_pairs == 'radiology':
            return len(self.cxr_files_all)
        elif self.cfg.dataset.data_pairs == 'partial_ehr_cxr':
            return len(self.ehr_files_paired) + int(self.data_ratio * len(self.ehr_files_unpaired)) 
        


def loadmetadata(cfg):

    data_dir = cfg.dataset.cxr_data_dir
    cxr_metadata = pd.read_csv(f'{data_dir}/mimic-cxr-2.0.0-metadata.csv')
    icu_stay_metadata = pd.read_csv(f'{cfg.dataset.ehr_data_dir}/root/all_stays.csv')
    
    # only common subjects with both icu stay and an xray
    cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata, how='inner', on='subject_id')
    
    # combine study date time
    cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
    cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")
    
    cxr_merged_icustays.intime=pd.to_datetime(cxr_merged_icustays.intime)
    cxr_merged_icustays.outtime=pd.to_datetime(cxr_merged_icustays.outtime)
    end_time = cxr_merged_icustays.outtime

    if cfg.dataset.task == 'in-hospital mortality':
        end_time = cxr_merged_icustays.intime + pd.DateOffset(hours=48)

    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&((cxr_merged_icustays.StudyDateTime<=end_time))]

    # select cxrs with the ViewPosition == 'AP
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']

    groups = cxr_merged_icustays_AP.groupby('stay_id')

    groups_selected = []
    for group in groups:
        # select the latest cxr for the icu stay
        selected = group[1].sort_values('StudyDateTime').tail(1).reset_index()
        groups_selected.append(selected)
    groups = pd.concat(groups_selected, ignore_index=True)

    return groups

# def 
def load_cxr_ehr(cfg):
    ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(cfg)
    cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(cfg)

    cxr_merged_icustays = loadmetadata(cfg) 

    splits_labels_train = pd.read_csv(f'{cfg.dataset.ehr_data_dir}/{cfg.dataset.task}/train_listfile.csv')
    splits_labels_val = pd.read_csv(f'{cfg.dataset.ehr_data_dir}/{cfg.dataset.task}/val_listfile.csv')
    splits_labels_test = pd.read_csv(f'{cfg.dataset.ehr_data_dir}/{cfg.dataset.task}/test_listfile.csv')


    train_meta_with_labels = cxr_merged_icustays.merge(splits_labels_train, how='inner', on='stay_id')
    val_meta_with_labels = cxr_merged_icustays.merge(splits_labels_val, how='inner', on='stay_id')
    test_meta_with_labels = cxr_merged_icustays.merge(splits_labels_test, how='inner', on='stay_id')
    
    train_ds = MIMIC_CXR_EHR(cfg, train_meta_with_labels, ehr_train_ds, cxr_train_ds)
    val_ds = MIMIC_CXR_EHR(cfg, val_meta_with_labels, ehr_val_ds, cxr_val_ds, split='val')
    test_ds = MIMIC_CXR_EHR(cfg, test_meta_with_labels, ehr_test_ds, cxr_test_ds, split='test')

    return train_ds, val_ds, test_ds

def my_collate(batch):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    img = torch.stack([torch.zeros(3, 384, 384) if item[1] is None else item[1] for item in batch])
    x, seq_length = pad_zeros(x)
    x, seq_length = torch.Tensor(x), torch.tensor(seq_length)
    targets_ehr = torch.Tensor(np.array([item[2] for item in batch])).unsqueeze(1)
    targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    meta_info = [item[4] for item in batch]
    age = torch.Tensor(np.array([item[5] for item in batch])).unsqueeze(1)
    gender = torch.Tensor(np.array([item[6] for item in batch]))
    ethnicity = torch.Tensor(np.array([item[7] for item in batch]))
    return {'ehr': x, 'img': img, 'targets_ehr': targets_ehr, 'targets_cxr': targets_cxr, 'seq_length': seq_length, 'pairs': pairs, 'meta_info': meta_info, 'age': age, 'gender': gender, 'ethnicity': ethnicity}

def pad_zeros(arr, min_length=None):
    # Pad with zeros until the every sample in batch has the same sequence length (maximal sequence length of every element)
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length


class MIMICCXR(Dataset):
    def __init__(self, paths, cfg, transform=None, split='train'):
        self.data_dir = cfg.dataset.cxr_data_dir
        self.cfg = cfg
        self.CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']
        self.filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}

        metadata = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-metadata.csv')
        labels = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-chexpert.csv')
        labels[self.CLASSES] = labels[self.CLASSES].fillna(0)
        labels = labels.replace(-1.0, 0.0)
        
        splits = pd.read_csv(f'{self.data_dir}/mimic-cxr-ehr-split.csv')


        metadata_with_labels = metadata.merge(labels[self.CLASSES+['study_id'] ], how='inner', on='study_id')


        self.filesnames_to_labels = dict(zip(metadata_with_labels['dicom_id'].values, metadata_with_labels[self.CLASSES].values))
        self.filenames_loaded = splits.loc[splits.split==split]['dicom_id'].values
        self.transform = transform
        self.filenames_loaded = [filename  for filename in self.filenames_loaded if filename in self.filesnames_to_labels]

    def __getitem__(self, index):
        if isinstance(index, str):
            img = Image.open(self.filenames_to_path[index]).convert('RGB')
            labels = torch.tensor(self.filesnames_to_labels[index]).float()

            if self.transform is not None:
                img = self.transform(img)
            return img, labels
        
        filename = self.filenames_loaded[index]
        
        img = Image.open(self.filenames_to_path[filename]).convert('RGB')

        labels = torch.tensor(self.filesnames_to_labels[filename]).float()

        if self.transform is not None:
            img = self.transform(img)
        return img, labels
    
    def __len__(self):
        return len(self.filenames_loaded)


def get_transforms(cfg):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transforms = []
    train_transforms.append(transforms.Resize(384))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(384))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)      


    test_transforms = []
    test_transforms.append(transforms.Resize(cfg.dataset.transforms.resize))


    test_transforms.append(transforms.CenterCrop(cfg.dataset.transforms.crop))

    test_transforms.append(transforms.ToTensor())
    test_transforms.append(normalize)


    return train_transforms, test_transforms

def get_cxr_datasets(cfg):
    train_transforms, test_transforms = get_transforms(cfg)

    data_dir = cfg.dataset.cxr_data_dir
    
    paths = glob.glob(f'{data_dir}/mimic-cxr-jpg/resized/**/*.jpg', recursive = True)
    
    dataset_train = MIMICCXR(paths, cfg, split='train', transform=transforms.Compose(train_transforms))
    dataset_validate = MIMICCXR(paths, cfg, split='validate', transform=transforms.Compose(test_transforms),)
    dataset_test = MIMICCXR(paths, cfg, split='test', transform=transforms.Compose(test_transforms),)

    return dataset_train, dataset_validate, dataset_test



def read_timeseries():
    path = f'/data/home/firas/Desktop/work/other_groups/MedFuse/mimic4extract/data/in-hospital-mortality/train/16662316_episode8_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)


class EHRdataset(Dataset):
    def __init__(self, listfile, dataset_dir, return_names=True, period_length=48.0):
        self.return_names = return_names
        self.discretizer = Discretizer(timestep = 1.0, store_masks=True, impute_strategy='previous', start_time='zero',
            config_path='/data/home/firas/Desktop/work/combine_image_and_text/classification/datasets/med_lab_utils/discretizer_config.json')

        discretizer_header = self.discretizer.transform(read_timeseries())[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
        self.normalizer = Normalizer(fields=cont_channels)
        self.normalizer.load_params('/data/home/firas/Desktop/work/combine_image_and_text/classification/datasets/med_lab_utils/ph_ts1.0.input_str:previous.start_time:zero.normalizer')

        self._period_length = period_length
        self._dataset_dir = dataset_dir

        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]

        self.CLASSES = self._listfile_header.strip().split(',')[3:]
        self._data = self._data[1:]

        self._data = [line.split(',') for line in self._data]
        self.data_map = {
            mas[0]: {
                'labels': list(map(float, mas[3:])),
                'stay_id': float(mas[2]),
                'time': float(mas[1]),
                }
                for mas in self._data
        }

        self.names = list(self.data_map.keys())
    
    def _read_timeseries(self, ts_filename, time_bound=None):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    
    def read_by_file_name(self, index, time_bound=None):
        t = self.data_map[index]['time'] if time_bound is None else time_bound
        y = self.data_map[index]['labels']
        stay_id = self.data_map[index]['stay_id']
        (X, header) = self._read_timeseries(index, time_bound=time_bound)

        return {"X": X,
                "t": t,
                "y": y,
                'stay_id': stay_id,
                "header": header,
                "name": index}

    def get_decomp_los(self, index, time_bound=None):
        return self.__getitem__(index, time_bound)

    def __getitem__(self, index, time_bound=None):
        if isinstance(index, int):
            index = self.names[index]
        ret = self.read_by_file_name(index, time_bound)
        data = ret["X"]
        ts = ret["t"] if ret['t'] > 0.0 else self._period_length
        ys = ret["y"]
        names = ret["name"]
        data = self.discretizer.transform(data, end=ts)[0] 
        if (self.normalizer is not None):
            data = self.normalizer.transform(data)
        ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        return data, ys

    
    def __len__(self):
        return len(self.names)


def get_datasets(cfg):
    train_ds = EHRdataset(f'{cfg.dataset.ehr_data_dir}/{cfg.dataset.task}/train_listfile.csv', os.path.join(cfg.dataset.ehr_data_dir, f'{cfg.dataset.task}/train'))
    val_ds = EHRdataset(f'{cfg.dataset.ehr_data_dir}/{cfg.dataset.task}/val_listfile.csv', os.path.join(cfg.dataset.ehr_data_dir, f'{cfg.dataset.task}/train'))
    test_ds = EHRdataset(f'{cfg.dataset.ehr_data_dir}/{cfg.dataset.task}/test_listfile.csv', os.path.join(cfg.dataset.ehr_data_dir, f'{cfg.dataset.task}/test'))
    return train_ds, val_ds, test_ds

@hydra.main(config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    train_dl, val_dl, test_dl = load_cxr_ehr(cfg)
    sample_ = next(iter(train_dl))
    print("End")

if __name__ == '__main__':
    run()
