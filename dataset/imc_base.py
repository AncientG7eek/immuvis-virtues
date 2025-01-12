import torch
import numpy as np
import torch
import random 
import os
import re
import pandas as pd
from einops import rearrange
from tqdm import tqdm
from utils.esm_utils import load_esm_dict
from sklearn.preprocessing import LabelEncoder
from utils.transform_utils import get_normalization_transform, CustomGaussianBlur, DropChannels, RandomRotation90, generate_mask, CropToPatchSize, GridReshape, MultiImageRandomCrop
from torchvision.transforms import v2
from sklearn.model_selection import GroupShuffleSplit  
from loguru import logger 
from torch.utils.data import Dataset

class IMCDataset:

    def __init__(self, conf,
                preload=False,
                filter_channel_names=None,
                filter_channel_indices=None,
            ):
        """
        A dataset consisting of images. 
        
        Args:
            conf: OmegaConfig object containing the configuration
            preload: If True, all images are preloaded into memory, defaults to False
            filter_channel_names: List of gene names to be used. If None, all genes are used. Defaults to None.
            filter_channel_indices: List of gene indices to be used. If None, all genes are used. Defaults to None.
        """
        # basic info, loading
        self.name = conf.dataset.name
        self.esm_model = conf.esm.name
        self.normalization = conf.image_info.normalization
        self.folder = f'{conf.dataset.path}/{self.name}'
        self.metadata_folder = f'{conf.dataset.metadata_path}/{self.name}'
        self.img_folder = os.path.join(self.folder, "images")
        self.mask_folder = os.path.join(self.folder, "masks")
        self.preprocessed_img_folder = os.path.join(self.folder, f"images_preprocessed_{self.normalization}")
        self._create_folder_if_not_exists(self.preprocessed_img_folder)
        self.preload = preload
        
        # crop info
        self.rnd_crop_size = conf.image_info.rnd_crop_size
        self.rnd_crop_folder = os.path.join(self.folder, f"random_crops_{self.normalization}_{self.rnd_crop_size}")
        self.rnd_crop_per_image = {}
        self._create_folder_if_not_exists(self.rnd_crop_folder)

        # load dictionary mapping protein id to index in the ESM embeddings matrix
        self.protein_id_to_index = load_esm_dict(conf) 

         # load gene to protein id dictionary
        self.gene_to_protein_id_dict_name = f'{self.metadata_folder}/gene_dict_{conf.dataset.name}.csv'
        self.gene_to_protein_id = {}
        assert self.gene_to_protein_id_dict_name.endswith(".csv"), "gene_to_protein_id_dict must be a csv file"
        self.genes = pd.read_csv(self.gene_to_protein_id_dict_name)
        self.gene_to_protein_id = dict(zip(self.genes['name'], self.genes['protein_id']))

        # Given a raw image with X channels, we want to use only a subset of them. First we take a subset of C channels for which the ESM embeddings of the protein are available. 
        # We call these "available". Then we potentially filter some more out. The call the remaining ones "used".
 
        self.protein_indices = [] # protein indices contain for each available channel its protein marker index in the ESM embeddings matrix
        self.gene_names = [] # gene names contain the names of gene marker of available channels
        self.gene_indices = [] # gene indices contain the indices of the available channels with respect to the full image.

        for i, row in self.genes.iterrows():
            protein_id = row["protein_id"]
            if protein_id in self.protein_id_to_index:
                self.gene_names.append(row["name"]) 
                self.gene_indices.append(i) 
                self.protein_indices.append(self.protein_id_to_index[protein_id])
        self.protein_indices = torch.tensor(self.protein_indices, dtype=torch.long) 
      
        self.filter_indices = None # fliter indices contain the indices of the channels that will be used, i.e. image_used = image[self.filter_indices]. None means all
        if filter_channel_names is not None:
            self.filter_indices = torch.tensor([self.gene_names.index(name) for name in filter_channel_names], dtype=torch.long)
        if filter_channel_indices is not None:
            self.filter_indices = torch.tensor(filter_channel_indices, dtype=torch.long)

        # load cell annotations
        self.annotation_path = f'{self.folder}/sce_annotations.csv'
        self.annotations = pd.read_csv(self.annotation_path, index_col=0)

        os.makedirs(f'dumps/{conf.dataset.name}', exist_ok=True)
        image_index_path = f'dumps/{conf.dataset.name}/patient_wise_image_index.csv'
        if os.path.exists(image_index_path):
            self.image_index = pd.read_csv(image_index_path, index_col=0)
            self.image_index["image_name"] = [str(img) for img in self.image_index["image_name"]]

        else:
            logger.info("Computing image index table for the first time. This may take a while...")
            self.image_index = self.create_image_index()
            self.image_index["image_name"] = [str(img) for img in self.image_index["image_name"]]
            self.image_index.to_csv(image_index_path)
        self.image_index = self.filter_image_index(self.image_index)

        self.preprocess_transform = v2.Compose([
            CustomGaussianBlur(kernel_size=3, sigma=1.0),
            get_normalization_transform(self.normalization),
        ])
        
        if self.preload:
            logger.info("Preloading images...")
            self.images = {}
            self.preload_images()
            logger.info("Images preloaded.")

        self.patch_level_tasks = []
        self.crop_level_tasks = []
        self.image_level_tasks = []

        self.cell_label_encoders = dict()
        self.cell_type_masks_buffer = dict()

    def get_gene_name_from_protein_index(self, protein_index):
        """
        Returns the gene name of a protein index.
        Args:
            protein_index: Index of the protein in the ESM embeddings matrix
        Returns:
            name of the gene
        """
        protein_id = list(self.protein_id_to_index.keys())[list(self.protein_id_to_index.values()).index(protein_index)]
        gene_name = self.genes[self.genes["protein_id"] == protein_id]["name"].values[0]
        return gene_name

    def restrict_image_index_to(self, column_name, allowed_values):
        """
        Applies a filter to the internal image index of the dataset. This can be used to exclude images at runtime.
        Args:
            column_name: Name of the column to filter
            allowed_values: List of allowed values
        """
        self.image_index = self.image_index[self.image_index[column_name].isin(allowed_values)]

    def get_protein_indices(self, image_id):
        """
        Returns indices of the protein markers measured in the image
        Args:
            image_id: Name of the image
        Returns:
            tensor of protein indices
        """
        if self.filter_indices is not None:
            return self.protein_indices[self.filter_indices]
        else:
            return self.protein_indices
    
    def get_sc_annotations(self, image_name):
        """
        Returns the single cell annotations for an image.
        Args:
            image_name: Name of the image
        Returns:
            pandas dataframe containing the annotations
        """
        sc_annotations = self.annotations[self.annotations["image_name"] == image_name]
        return sc_annotations
    
    def load_gene_rates(self, image_id : str, flatten : bool = False, preprocess : bool = True, channel_first : bool = False):
        """
        Loads the expression rates an image.
        Args:
            image_id: Name of the image
            flatten: If True, the spatial dimensions of the image are flattened. Defaults to False.
            preprocess: If True, the loads the preprocessed image is preprocessed. Defaults to True.
            channel_first: If True, the channels are the first dimension. Defaults to False.
        Returns:
            tensor containing the image
        """
        if self.preload:
            rates = self.images[image_id] # c x h x w
        else:
            if preprocess:
                if self._is_available_preprocessed(image_id):
                    rates = self._load_preprocessed_image(image_id)
                else:
                    path = os.path.join(self.img_folder, image_id + ".npy")
                    rates = np.load(path)
                    rates = self.preprocess_image(rates)
                    self._save_preprocessed_image(image_id, rates)
                    logger.info(f"Saving image {image_id} as preprocessed with normalization {self.normalization} for the first time")
            else:
                path = os.path.join(self.img_folder, image_id + ".npy")
                rates = np.load(path) # c x h x w
            rates = rates[self.gene_indices]
            rates = torch.tensor(rates).float()
            if self.filter_indices is not None:
                rates = rates[self.filter_indices]
        if channel_first:
            rates  = rearrange(rates, 'c h w -> c (h w)') if flatten else rates
        else:
            rates = rearrange(rates, 'c h w -> (h w) c') if flatten else rearrange(rates, 'c h w -> h w c')
        return rates
    
    def create_random_crops_dir(self):
        """
        Creates a directory with random crops of the images. This is useful for training with reduced I/O usage.
        """
        temp_filter = self.filter_indices
        self.filter_indices = None

        for image_id in tqdm(self.get_image_names()):
            image  = self.load_gene_rates(image_id, flatten=False, preprocess=True, channel_first=True)
            H, W = image.shape[1], image.shape[2]
            max_crops = (H / self.rnd_crop_size) * (W / self.rnd_crop_size)
            max_crops  = int(max_crops)*4
            for i in range(max_crops):
                x = random.randint(0, W - self.rnd_crop_size)
                y = random.randint(0, H - self.rnd_crop_size)
                crop = image[:, y:y+self.rnd_crop_size, x:x+self.rnd_crop_size]
                crop = crop.numpy()
                np.save(os.path.join(self.rnd_crop_folder, f"{image_id}_{i}.npy"), crop)
        
        self.filter_indices = temp_filter
    
    def load_random_crop(self, image_id):
        """
        Loads a random crop for the specified image.
        Args:
            image_id: Name of the image
        """
        if image_id in self.rnd_crop_per_image:
            num_crops = self.rnd_crop_per_image[image_id]
        else:
            crop_files = os.listdir(self.rnd_crop_folder)
            crop_files = [f for f in crop_files if f.startswith(image_id + "_")]
            num_crops = len(crop_files)
            self.rnd_crop_per_image[image_id] = num_crops
        i = np.random.randint(0, num_crops)
        crop = np.load(os.path.join(self.rnd_crop_folder, f"{image_id}_{i}.npy"))
        crop = torch.tensor(crop).float()
        if self.filter_indices is not None:
            crop = crop[self.filter_indices]
        return crop

    def preload_images(self, preprocess=True):
        """
        Preloads all images into memory.
        Args:
            preprocess: If True, the images are preprocessed. Defaults to True.
        """
        for image_id in tqdm(self.get_image_names()):
            if preprocess:
                if self._is_available_preprocessed(image_id):
                    rates = self._load_preprocessed_image(image_id)
                else:
                    path = os.path.join(self.img_folder, image_id + ".npy")
                    rates = np.load(path)
                    rates = self.preprocess_image(rates)
                    self._save_preprocessed_image(image_id, rates)
                    logger.info(f"Saving image {image_id} as preprocessed with normalization {self.normalization} for the first time")
            else:
                path = os.path.join(self.img_folder, image_id + ".npy")
                rates = np.load(path) # c x h x w
            rates = rates[self.gene_indices]
            rates = torch.tensor(rates).float()
            if self.filter_indices is not None:
                rates = rates[self.filter_indices]
            self.images[image_id] = rates

    def __len__(self):  
        return len(self.get_image_names())

    def __getitem__(self, idx):
        image_id = self.get_image_names()[idx]
        image = self.load_gene_rates(image_id, flatten=False, preprocess=True, channel_first=True)
        return image

    def load_mask(self, mask_id: str, flatten: bool = False):
        """
        Loads cell segmentation mask.
        Args:
            mask_id: Name of the mask
            flatten: If True, the spatial dimensions of the mask are flattened. Defaults to False.
        Returns:
            tensor containing the mask
        """
        path = os.path.join(self.mask_folder, mask_id + ".npy")
        mask = np.load(path)
        mask = mask.astype(int)
        mask = torch.asarray(mask)
        if flatten:
            return mask.flatten()
        else:
            return mask
       
    def filter_image_index(self, image_index):
        """
        Filters the image index according to some criterion (to be overwriten)
        """
        image_index  = image_index[image_index["width"] >= self.rnd_crop_size]
        image_index = image_index[image_index["height"] >= self.rnd_crop_size]
        return image_index

    def create_image_index(self, split=0.8):
        """
        Creates an index table of images split into training and testing images. Split is grouped by 'patient_id' if available.
        """
        clinical = pd.read_csv(f'{self.folder}/clinical.csv')

        if "patient_id" in clinical.columns:
            groups = clinical["patient_id"]
        else:
            logger.info("No patient_id column found in clinical data. Splitting randomly.")
            groups = range(len(clinical))

        _splitter = GroupShuffleSplit(n_splits=1, test_size=1-split, random_state=42)
        _split = _splitter.split(clinical, groups=groups)
        train_idx, test_idx = next(_split)
        
        clinical['split'] = 'train'
        clinical.loc[test_idx, 'split'] = 'test'

        if "patient_id" in clinical.columns:
            logger.info(f'Train contains {len(train_idx)} images from {len(clinical[clinical["split"] == "train"]["patient_id"].unique())} patients')
            logger.info(f'Test contains {len(test_idx)} images from {len(clinical[clinical["split"] == "test"]["patient_id"].unique())} patients')

        else:
            logger.info(f'Train contains {len(train_idx)} images')
            logger.info(f'Test contains {len(test_idx)} images')

        heights = []
        widths = []

        for img in tqdm(clinical["image_name"]):
            path = os.path.join(self.img_folder, img + ".npy")
            shape = np.load(path).shape

            height = shape[1]
            width = shape[2]

            heights.append(height)
            widths.append(width) 

        clinical["height"] = heights
        clinical["width"] = widths

        return clinical

    def get_image_index(self, split=None):
        """
        Returns index table of images split into training and testing images
        Args:
            split: Split of the dataset to use. Defaults to None which returns the full image index.
        """
        if split is None:
            return self.image_index
        else:
            return self.image_index[self.image_index["split"] == split]
    
    def get_image_names(self, split=None):
        """
        Returns a list of image names
        """
        return self.get_image_index(split=split)["image_name"].values
    
    def get_max_grid_size(self, patch_size, stride=0):
        """
        Returns the maximum grid size for the given patch size
        """
        max_grid_size = 0

        for w, h in zip(self.get_image_index()["width"], self.get_image_index()["height"]):
            grid_size = ((w // (patch_size+stride)) - 1 ) * ((h // (patch_size+stride)) - 1)
            max_grid_size = max(max_grid_size, grid_size)
            
        return max_grid_size
    
    def get_min_grid_size(self, patch_size, stride=0):
        """
        Returns the maximum grid size for the given patch size
        """
        min_grid_size = 1000000

        for w, h in zip(self.get_image_index()["width"], self.get_image_index()["height"]):
            grid_size = ((w // (patch_size+stride)) - 1 ) * ((h // (patch_size+stride)) - 1)
            min_grid_size = min(min_grid_size, grid_size)
        
        return min_grid_size

    def get_num_crops(self, crop_size):
        """
        Computes the number of distinc non-overlapping crops of size crop_size x crop_size that can be extracted from the images.
        """
        count = 0
        for w, h in zip(self.get_image_index()["width"], self.get_image_index()["height"]):
            count += (w // crop_size) * (h // crop_size)
        return count

    def preprocess_image(self, image):
        """
        Preprocesses an image by applying 99th percentile clipping and log-normalization, a gaussian blur and normalization.
        Args:
            image: numpy array of shape (channels, height, width)
        Returns
            preprocessed image
        """
        image = np.clip(image, 0, np.percentile(image, 99, axis=(1,2), keepdims=True)).astype(np.float32)
        image = np.log1p(image) # log1p = log(1 + x)
        image = self.preprocess_transform(image)
        return image

    def _is_available_preprocessed(self, image_id : str):
        path = os.path.join(self.preprocessed_img_folder, image_id + ".npy")
        return os.path.exists(path)

    def _load_preprocessed_image(self, image_id : str):
        path = os.path.join(self.preprocessed_img_folder, image_id + ".npy")
        return np.load(path)

    def _save_preprocessed_image(self, image_id : str, image):
        path = os.path.join(self.preprocessed_img_folder, image_id + ".npy")
        np.save(path, image)

    def _create_folder_if_not_exists(self, path : str):
        if not os.path.exists(path):
            os.makedirs(path)

    def setup_cell_label_encoder(self, column_name="cell_category"):
        """
        Sets up a label encoder for the cell type annotations.
        Args:
            column_name: Name of the column containing the cell type annotations.
        """
        if self.cell_label_encoders.get(column_name) is None:
            le = LabelEncoder()
            names = self.annotations[column_name].unique().tolist()
            names.append("None")
            le.fit(names)
            self.cell_label_encoders[column_name] = le

    def load_cell_type_mask(self, mask_id, column_name="cell_category"):
        """
        Loads the cell type mask.
        Args: 
            column_name: Name of the column in single-cell annotations containing the cell types.
        """
        if self.cell_type_masks_buffer.get(column_name) is not None:
            if self.cell_type_masks_buffer[column_name].get(mask_id) is not None:
                return self.cell_type_masks_buffer[column_name][mask_id]
        else:
            self.cell_type_masks_buffer[column_name] = {}
        self.setup_cell_label_encoder(column_name)
        mask = self.load_mask(mask_id)
        sc_ann = self.get_sc_annotations(mask_id)

        mapping_dict = dict(zip(sc_ann["cell_id"], self.cell_label_encoders[column_name].transform(sc_ann[column_name])))
        mapping_dict[0] = self.cell_label_encoders[column_name].transform(["None"])[0]
        func = lambda x: mapping_dict.get(x, mapping_dict[0])
        mask = mask.apply_(func)
        self.cell_type_masks_buffer[column_name][mask_id] = mask
        return mask
    
    def encode_cell_type(self, cell_type_name, column_name="cell_category"):
        """
        Encodes the name of a cell type into its class id. 
        Args:
            cell_type_name: Name of the cell type or array of such.
            column_name: Name of the column in single-cell annotations containing the cell types.
        """
        self.setup_cell_label_encoder(column_name)
        if isinstance(cell_type_name, str):
            return self.cell_label_encoders[column_name].transform([cell_type_name])[0]
        else:
            return self.cell_label_encoders[column_name].transform(cell_type_name)

    def decode_cell_type(self, cell_type_id, column_name="cell_category"):
        """
        Decodes the class id of a cell type into its name.
        Args:
            cell_type_id: Class id of the cell type or array of such.
            column_name: Name of the column in single-cell annotations containing the cell types.
        """
        assert self.cell_label_encoders.get(column_name) is not None, "Cell label encoder not found which should not be the case because how did you obtain the cell_type_id in the first place?"
        if isinstance(cell_type_id, int):
            return self.cell_label_encoders[column_name].inverse_transform([cell_type_id])[0]
        else:
            return self.cell_label_encoders[column_name].inverse_transform(cell_type_id)

    def cell_type_to_id(self, column_name="cell_category"):
        """
        Returns a dictionary mapping cell type names to their class id.
        Args:
            column_name: Name of the column in single-cell annotations containing the cell
        """
        self.setup_cell_label_encoder(column_name)
        return dict(zip(self.cell_label_encoders[column_name].classes_, self.cell_label_encoders[column_name].transform(self.cell_label_encoders[column_name].classes_)))

    def id_to_cell_type(self, column_name="cell_category"):
        """
        Returns a dictionary mapping cell type class id to their name.
        Args:
            column_name: Name of the column in single-cell annotations containing the cell types.
        """
        self.setup_cell_label_encoder(column_name)
        return dict(zip(self.cell_label_encoders[column_name].transform(self.cell_label_encoders[column_name].classes_), self.cell_label_encoders[column_name].classes_))   


class UnionIMCDataset(IMCDataset):

    def __init__(self, conf, name, datasets_dict: dict[str, IMCDataset]):
        """
        Wrapper class to combine multiple IMCDatasets into one.
        Args:
            conf: OmegaConfig object containing the configuration
            name: Name of the dataset
            datasets_dict: Dictionary containing the datasets with their name as the key
        """
        self.name = name

        self.datasets_dict = datasets_dict

        for ds_name in self.datasets_dict.keys():
            assert self.datasets_dict[ds_name].esm_model == conf.esm.name, "All datasets in UnionIMCDataset must have the same esm_model"

        self.image_index = []

        for ds_name in self.datasets_dict.keys():
            img_index = self.datasets_dict[ds_name].image_index.copy(deep=True)
            img_index["dataset_name"] = ds_name
            img_index["image_name"] = img_index["dataset_name"] + "_" + img_index["image_name"].astype("string")
            self.image_index.append(img_index)
            
        self.image_index = pd.concat(self.image_index, ignore_index=True)

    def __len__(self):
        return sum([len(ds) for ds in self.datasets_dict.values()])
    
    def __getitem__(self, idx):
        for ds_name in self.datasets_dict.keys():
            if idx < len(self.datasets_dict[ds_name]):
                return self.datasets_dict[ds_name].__getitem__(idx)
            idx -= len(self.datasets_dict[ds_name])
        raise IndexError("Index out of bounds")

    def get_protein_indices(self, image_id):
        splits = re.split(r'_', image_id, maxsplit=1)
        dataset_name = splits[0]
        image_name = splits[1]
        return self.datasets_dict[dataset_name].get_protein_indices(image_name)
    
    def load_gene_rates(self, image_id : str, flatten : bool =True, preprocess : bool = True, channel_first : bool = False):
        # Split image name into original image name and dataset
        splits = re.split(r'_', image_id, maxsplit=1)
        dataset_name = splits[0]
        image_name = splits[1]
        return self.datasets_dict[dataset_name].load_gene_rates(image_name, flatten=flatten, preprocess=preprocess, channel_first=channel_first)

    def create_random_crops_dir(self):
        for ds_name in self.datasets_dict.keys():
            self.datasets_dict[ds_name].create_random_crops_dir()

    def load_random_crop(self, image_id):
        splits = re.split(r'_', image_id, maxsplit=1)
        dataset_name = splits[0]
        image_name = splits[1]
        return self.datasets_dict[dataset_name].load_random_crop(image_name)

    def load_mask(self, mask_id: str, flatten: bool = False):
        # Split image name into original image name and dataset
        splits = re.split(r'_', mask_id, maxsplit=1)
        dataset_name = splits[0]
        mask_name = splits[1]
        return self.datasets_dict[dataset_name].load_mask(mask_name, flatten=flatten)

    def get_dir_tokens_training(self, conf):
        raise NotImplemented("Cannot call this method on Union IMCDataset")
    
    def get_dir_transformer_training(self, conf):
        raise NotImplemented("Cannot call this method on Union IMCDataset")
    
    def create_image_index(self, split=0.8):
        raise NotImplemented("Cannot call this method on Union IMCDataset")
    
    def setup_cell_label_encoder(self, column_name="cell_category"):
        raise NotImplemented("Cannot call this method on Union IMCDataset")
    
    def load_cell_type_mask(self, mask_id, column_name="cell_category"):
        raise NotImplemented("Cannot call this method on Union IMCDataset")
    
    def encode_cell_type(self, cell_type_name, column_name="cell_category"):
        raise NotImplemented("Cannot call this method on Union IMCDataset")
    
    def decode_cell_type(self, cell_type_id, column_name="cell_category"):
        raise NotImplemented("Cannot call this method on Union IMCDataset")
    
    def cell_type_to_id(self, column_name="cell_category"):
        raise NotImplemented("Cannot call this method on Union IMCDataset")
    
    def id_to_cell_type(self, column_name="cell_category"):
        raise NotImplemented("Cannot call this method on Union IMCDataset")

class IMCImageDataset(Dataset):

    def __init__(self, conf, imcdataset, split='train'):
        self.split = split
        self.imc_dataset = imcdataset
        self.image_names = self.imc_dataset.get_image_names(split=self.split)
        self.patch_size = conf.image_info.patch_size
        self.evaluation_mode = False
        self.return_metadata = False
        
        self.cell_type_column = "cell_category"
        self.use_rnd_crop_dir = conf.image_info.use_rnd_crop_dir if split == 'train' else False

        self.use_fraction_training = conf.image_info.use_fraction_training

    def get_image_index(self):
        return self.imc_dataset.get_image_index(split=self.split)

    def get_metadata(self, idx):
        metadata = {}
        metadata["image_name"] = self.image_names[idx]
        return metadata

    def __len__(self):
        return int(len(self.image_names)*self.use_fraction_training)

    def __getitem__(self, idx):
        if self.use_rnd_crop_dir:
            img = self.imc_dataset.load_random_crop(self.image_names[idx])
        else:
            img = self.imc_dataset.load_gene_rates(self.image_names[idx], flatten=False, preprocess=True, channel_first=True)
        if self.return_metadata:
            metadata = self.get_metadata(idx)
            return img, metadata
        else:
            return img


class MAEDataset(IMCImageDataset):

    def __init__(self, conf, imcdataset, split='train'):
        """
        Dataset for training a MAE model.
        """
        super().__init__(conf, imcdataset, split=split)
        self.mask_ratio = conf.image_info.mask_ratio
        self.mask_strategy = conf.image_info.mask_strategy

        self.channel_fraction = conf.image_info.channel_fraction

        self.preselection_transform = v2.Compose(
            [
                v2.RandomCrop(size=conf.image_info.image_section_size, pad_if_needed=True)
            ]
        )
        self.augmentation_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                RandomRotation90(p=0.5)
            ]
        )

        self.drop_channels = DropChannels(p=1.0, fraction_range=self.channel_fraction)

        self.crop_to_patchsize = CropToPatchSize(patch_size=self.patch_size)
        self.to_grid = GridReshape(patch_size=self.patch_size)

    def __getitem__(self, idx):
        """
        Returns a training sample consisting of image, protein indices and mask.
        """
        img = super().__getitem__(idx)

        img = self.preselection_transform(img)
        img = self.augmentation_transform(img)

        channels = self.imc_dataset.get_protein_indices(self.image_names[idx])

        img, channels = self.drop_channels(img, channels)

        img = self.crop_to_patchsize(img)
        img = self.to_grid(img)
        C, H, W = img.shape[:-1]
        mask = generate_mask(C, H, W, self.mask_ratio, mask_strategy=self.mask_strategy)

        return img, channels, mask
    

class ChannelAgnosticMAEDataset(IMCImageDataset):
    def __init__(self, conf, imc_dataset, split):
        super().__init__(conf, imc_dataset, split=split)
        self.image_section_size = conf.image_info.image_section_size
        self.mask_ratio = conf.image_info.mask_ratio
        self.preselection_transform = v2.Compose(
            [
                v2.RandomCrop(size=self.image_section_size, pad_if_needed=True)
            ]
        )

        self.augmentation_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )

    def __getitem__(self, idx):
        img = super().__getitem__(idx)
        img = self.preselection_transform(img)
        img = self.augmentation_transform(img)

        return img
    

class ImageEvalDataset(IMCImageDataset):

    def __init__(self, conf, imcdataset, image_section_size=(256, 256), stride=None, patch_size=16, split="train", grid_reshape=True):
        """
        Dataset for evaluating image-level tasks.
        Args:
            conf: OmegaConfig object containing the configuration
            imcdataset: IMCDataset object
            image_section_size: Size of the image sections
            stride: Stride for cropping the image. If none, uses image section size as stride.
            patch_size: Size of the patches
            split: Split of the dataset to use. Either 'train' or 'test'. Defaults to 'train'.
            grid_reshape: If True, reshapes the image into a grid, i.e. to the shape grid_height x grid_width x (patch_size**2). Defaults to True.
        """
        super().__init__(conf, imcdataset, split)

        self.return_metadata = True
        self.image_section_size = image_section_size
        self.stride = stride

        self.crop_to_patchsize = CropToPatchSize(patch_size=self.patch_size)
        self.to_grid = GridReshape(patch_size=self.patch_size)

        self.grid_reshape = grid_reshape

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        """
        Returns batch of all crops within an image, protein indices and metadata containing shape of grid of crops and registered task labels.
        """
        img, metadata =  super().__getitem__(idx)
        
        # cell_mask = self.imc_dataset.load_cell_type_mask(metadata["image_name"], column_name="cell_category") # h x w

        crops = []

        h, w = img.shape[-2:]
        if self.stride is None:
            stride_h = self.image_section_size[0]
            stride_w = self.image_section_size[1]
        else:
            stride_h = self.stride
            stride_w = self.stride

        nr = (h - self.image_section_size[0]) // stride_h + 1
        nc = (w - self.image_section_size[1]) // stride_w + 1

        for i in range(nr):
            for j in range(nc):
                img_crop = img[..., i*stride_h:i*stride_h+self.image_section_size[0], j*stride_w:j*stride_w+self.image_section_size[1]]
                if self.grid_reshape:
                    img_crop = self.crop_to_patchsize(img_crop)
                    img_crop = self.to_grid(img_crop)
                crops.append(img_crop)

        crops = torch.stack(crops, dim=0)

        # cell_mask = cell_mask[:(nr-1)*stride_h+self.image_section_size[0], :(nc-1)*stride_w+self.image_section_size[1]]
        # metadata["cell_mask"] = cell_mask
        
        metadata["num_rows"] = nr
        metadata["num_columns"] = nc
        metadata["num_crops"] = nr*nc

        for i_task in self.imc_dataset.image_level_tasks:
            metadata[i_task.task_name] = i_task.get_label(metadata["image_name"])
        
        channels = self.imc_dataset.get_protein_indices(self.image_names[idx]).expand(crops.shape[0], -1)

        return crops, channels, metadata
    

class CropEvalDataset(IMCImageDataset):

    def __init__(self, conf, imcdataset, image_section_size=(256, 256), patch_size=16, split="train", grid_reshape=True):
        """
        Dataset for evaluating image-level tasks.
        Args:
            conf: OmegaConfig object containing the configuration
            imcdataset: IMCDataset object
            image_section_size: Size of the image sections
            stride: Stride for cropping the image. If none, uses image section size as stride.
            patch_size: Size of the patches
            split: Split of the dataset to use. Either 'train' or 'test'. Defaults to 'train'.
            grid_reshape: If True, reshapes the image into a grid, i.e. to the shape grid_height x grid_width x (patch_size**2). Defaults to True.
        """
        super().__init__(conf, imcdataset, split)

        self.return_metadata = True
        self.image_section_size = image_section_size

        self.grid_reshape = grid_reshape
        self.crop_to_patchsize = CropToPatchSize(patch_size=self.patch_size)

        self.to_grid = GridReshape(patch_size=self.patch_size)


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Returns batch of all crops within an image, protein indices and metadata containing shape of grid of crops and registered task labels per crop.
        """
        img, metadata = super().__getitem__(idx)

        masks = {}
        for c_task in self.imc_dataset.crop_level_tasks:
            masks[c_task.task_name] = c_task.get_full_mask(metadata["image_name"])

        h, w = img.shape[-2:]

        nr = h // self.image_section_size[0]
        nc = w // self.image_section_size[1]

        offset_h = h % self.image_section_size[0]
        offset_w = w % self.image_section_size[1]

        offset_h_top = offset_h // 2
        offset_w_left = offset_w // 2

        subimages = []
        submasks = {c_task.task_name: [] for c_task in self.imc_dataset.crop_level_tasks}

        for r in range(nr):
            for c in range(nc):
                x1 = offset_h_top + r*self.image_section_size[0]
                x2 = offset_h_top + (r+1)*self.image_section_size[0]
                y1 = offset_w_left + c*self.image_section_size[1]
                y2 = offset_w_left + (c+1)*self.image_section_size[1]
                
                subimg = img[:, x1:x2, y1:y2]
                subimages.append(subimg)
                
                for c_task in self.imc_dataset.crop_level_tasks:
                    submasks[c_task.task_name].append(masks[c_task.task_name][x1:x2, y1:y2])

        for i in range(len(subimages)):
            si = subimages[i]
            if self.grid_reshape:
                si = self.crop_to_patchsize(si)
                si = self.to_grid(si)
            subimages[i] = si

        subimages = torch.stack(subimages)
        metadata["num_subimages"] = len(subimages)

        for c_task in self.imc_dataset.crop_level_tasks:
            metadata[c_task.task_name] = torch.stack([c_task.mask_to_label(sm) for sm in submasks[c_task.task_name]], dim=0)
            
        channels = self.imc_dataset.get_protein_indices(self.image_names[idx]).expand(subimages.shape[0], -1)
        return subimages, channels, metadata
    

class PatchEvalDataset(IMCImageDataset):

    def __init__(self, conf, imcdataset, image_section_size=(256, 256), patch_size=16, split="train", grid_reshape=True):
        """
        Dataset for evaluating patch-level tasks.
        Args:
            conf: OmegaConfig object containing the configuration
            imcdataset: IMCDataset object
            image_section_size: Size of the image sections
            stride: Stride for cropping the image. If none, uses image section size as stride.
            patch_size: Size of the patches
            split: Split of the dataset to use. Either 'train' or 'test'. Defaults to 'train'.
            grid_reshape: If True, reshapes the image into a grid, i.e. to the shape grid_height x grid_width x (patch_size**2). Defaults to True.
        """
        super().__init__(conf, imcdataset, split)

        self.return_metadata = True

        self.crop_to_patchsize = CropToPatchSize(patch_size=self.patch_size)
        self.multi_image_random_crop = MultiImageRandomCrop(size=image_section_size, return_coordinates=True)

        self.grid_reshape = grid_reshape
        self.to_grid = GridReshape(patch_size=self.patch_size)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Returns crop, protein indices and metadata containing the row and column of the crop and registered task labels per patch.
        """
        img, metadata = super().__getitem__(idx)

        full_masks = []
        for p_task in self.imc_dataset.patch_level_tasks:
            full_masks.append(p_task.get_full_mask(metadata["image_name"]))

        output, (row, col) = self.multi_image_random_crop(img, *full_masks)
        img = output[0]
        masks = output[1:]


        metadata["row"] = row
        metadata["col"] = col

        img = self.crop_to_patchsize(img)
        masks = [self.crop_to_patchsize(m) for m in masks]
        if self.grid_reshape:
            img = self.to_grid(img)

        for p_task, mask in zip(self.imc_dataset.patch_level_tasks, masks):
            metadata[p_task.task_name] = p_task.mask_to_label_grid(mask, self.patch_size)

        channels = self.imc_dataset.get_protein_indices(self.image_names[idx])        

        return img, channels, metadata
    

class PatchEvalDatasetFixedCoordinates(IMCImageDataset):

    def __init__(self, conf, crop_coordinates, imcdataset, image_section_size=(256, 256), patch_size=16, split="train", grid_reshape=True):
        """
        Dataset for evaluating patch-level tasks using fixed patch and crop coordinates. This is intended to ensure fair comparison between different benchmarked models.
        Args:
            conf: OmegaConfig object containing the configuration
            imcdataset: IMCDataset object
            image_section_size: Size of the image sections
            stride: Stride for cropping the image. If none, uses image section size as stride.
            patch_size: Size of the patches
            split: Split of the dataset to use. Either 'train' or 'test'. Defaults to 'train'.
            grid_reshape: If True, reshapes the image into a grid, i.e. to the shape grid_height x grid_width x (patch_size**2). Defaults to True.
        """
        super().__init__(conf, imcdataset, split)

        self.return_metadata = True

        self.crop_to_patchsize = CropToPatchSize(patch_size=self.patch_size)
        self.image_section_size = image_section_size

        self.grid_reshape = grid_reshape
        self.to_grid = GridReshape(patch_size=self.patch_size)

        self.crop_coordinates = crop_coordinates
    def __len__(self):
        return len(self.crop_coordinates)

    def __getitem__(self, idx):
        """
        Returns crop, protein indices and metadata containing the row and column of the crop and registered task labels per patch.
        """
        coord = self.crop_coordinates[idx]
        img_name = coord['image_name']

        img = self.imc_dataset.load_gene_rates(img_name, flatten=False, preprocess=True, channel_first=True)
        metadata = {'image_name': coord['image_name']}
        patch_indices = coord['patch_indices']

        row = coord['row']
        col = coord['col']

        img = img[:, row:row+self.image_section_size[0], col:col+self.image_section_size[1]]

        full_masks = []
        for p_task in self.imc_dataset.patch_level_tasks:
            full_masks.append(p_task.get_full_mask(metadata["image_name"])[row:row+self.image_section_size[0], col:col+self.image_section_size[1]])

        img = self.crop_to_patchsize(img)
        masks = [self.crop_to_patchsize(m) for m in full_masks]

        if self.grid_reshape:
            img = self.to_grid(img)

        for p_task, mask in zip(self.imc_dataset.patch_level_tasks, masks):
            metadata[p_task.task_name] = p_task.mask_to_label_grid(mask, self.patch_size)

        channels = self.imc_dataset.get_protein_indices(img_name)        
        return img, channels, metadata, patch_indices
    

class CoordinateDumper(IMCImageDataset):
    def __init__(self, conf, imcdataset, image_section_size=(256, 256), patch_size=16, split="train", grid_reshape=True):
        """
        Helper Dataset class used to generate coordinates of crops and patches within.
        """
        super().__init__(conf, imcdataset, split)

        self.return_metadata = True

        self.crop_to_patchsize = CropToPatchSize(patch_size=self.patch_size)

        self.grid_reshape = grid_reshape
        self.to_grid = GridReshape(patch_size=self.patch_size)
        self.image_section_size = image_section_size

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Generates coordinates of crops and selects a subset of their patches.
        Returns:
            list of dictionaries containing the coordinates of the image name, the crops and the indices of the patches
        """
        img, metadata = super().__getitem__(idx)

        full_masks = []
        for p_task in self.imc_dataset.patch_level_tasks:
            full_masks.append(p_task.get_full_mask(metadata["image_name"]))

        h, w = img.shape[-2:]

        nr = h // self.image_section_size[0]
        nc = w // self.image_section_size[1]

        offset_h = h % self.image_section_size[0]
        offset_w = w % self.image_section_size[1]

        offset_h_top = offset_h // 2
        offset_w_left = offset_w // 2

        # subimages = []
        # submasks = {p_task.task_name: [] for p_task in self.imc_dataset.patch_level_tasks}
        crop_coordinates = []
        for r in range(nr):
            for c in range(nc):
                x1 = offset_h_top + r*self.image_section_size[0]
                x2 = offset_h_top + (r+1)*self.image_section_size[0]
                y1 = offset_w_left + c*self.image_section_size[1]
                y2 = offset_w_left + (c+1)*self.image_section_size[1]
                
                # subimg = img[:, x1:x2, y1:y2]
                # subimages.append(subimg)
                crop_coordinates.append(
                    {
                        'row': r,
                        'col': c,
                        'image_name': metadata["image_name"],
                        'patch_indices': np.array(list(range(16 * 16)))
                    }
                )
                
                # for idxss, p_task in enumerate(self.imc_dataset.patch_level_tasks):
                    # submasks[p_task.task_name].append(full_masks[idxss][x1:x2, y1:y2])

        return crop_coordinates

