from dataset.imc_base import IMCDataset, UnionIMCDataset
from utils.data_utils import PatchLevelMulticlassTask, ImageLevelMulticlassTask, CropLevelBinaryStructureTask
import os
from loguru import logger

def get_imc_dataset(conf, filter_channel_names=None, filter_channel_indices=None):
    """
    Get the IMC dataset object based on the configuration.
    Returns: IMCDataset object
    """
    if conf.dataset.name == "lung":
        return IMCDatasetCords(conf, preload=False, filter_channel_names=filter_channel_names, filter_channel_indices=filter_channel_indices)
    elif conf.dataset.name == "hochschulz":
        return IMCDatasetHochschulz(conf, preload=False, filter_channel_names=filter_channel_names, filter_channel_indices=filter_channel_indices)
    elif conf.dataset.name == "jacksonfischer":
        return IMCDatasetJacksonFischer(conf, preload=False, filter_channel_names=filter_channel_names, filter_channel_indices=filter_channel_indices)
    elif conf.dataset.name == "danenberg":
        return IMCDatasetDanenberg(conf, preload=False, filter_channel_names=filter_channel_names, filter_channel_indices=filter_channel_indices)
    elif conf.dataset.name == 'damond':
        return IMCDatasetDamond(conf, preload=False, filter_channel_names=filter_channel_names, filter_channel_indices=filter_channel_indices)
    else:
        if os.path.exists(f'{conf.dataset.path}/{conf.dataset.name}'):
            logger.info(f"Using custom dataset {conf.dataset.name}")
            return IMCDataset(conf, preload=False, filter_channel_names=filter_channel_names, filter_channel_indices=filter_channel_indices)
        else:
            raise ValueError(f"No dataset directory found for custom dataset {conf.dataset.name}")


def get_union_imc_datasets(conf, union_list, filter_channel_names=None, filter_channel_indices=None):
    """
    Returns a UnionIMCDataset object with multiple datasets.
    Args:
        conf: OmegaConf object with all the configuration
        union_list: list of dataset names to be combined
        filter_channel_names: list of channel names to be included in the dataset
        filter_channel_indices: list of protein indices to be included in the dataset
    """
    datasets = {}
    original_dataset_name = conf.dataset.name
    for dataset_name in union_list:
        conf.dataset.name = dataset_name
        datasets[dataset_name] = get_imc_dataset(conf, filter_channel_names=filter_channel_names, filter_channel_indices=filter_channel_indices)
    conf.dataset.name = original_dataset_name
    return UnionIMCDataset(conf, ' '.join(union_list), datasets)
    

class IMCDatasetDamond(IMCDataset):

    def __init__(self, conf, preload, filter_channel_names, filter_channel_indices):
        """
        IMCDataset class with tasks for Damond et. al publication.
        """
        super().__init__(conf, preload, filter_channel_names, filter_channel_indices)
        
        self.image_level_tasks = [
                ImageLevelMulticlassTask(self, "stage", column_name="stage", final_eval=False),
                ImageLevelMulticlassTask(self, "aab_status", column_name="aab_status", final_eval=False),
                ]
        
        self.patch_level_tasks = [
            PatchLevelMulticlassTask(self, "cell_category", column_name="cell_category"),
        ]

class IMCDatasetHochschulz(IMCDataset):
    def __init__(self, conf, preload, filter_channel_names, filter_channel_indices):
        """
        IMCDataset class with tasks for Hoch et. al publication.
        """
        super().__init__(conf, preload, filter_channel_names, filter_channel_indices)
        
        self.image_level_tasks = [
                ImageLevelMulticlassTask(self, "patient_cancer_stage", column_name="patient_cancer_stage", final_eval=True),
                ImageLevelMulticlassTask(self, "patient_relapse", column_name="patient_relapse", final_eval=True),
                ImageLevelMulticlassTask(self, "patient_mutation", column_name="patient_mutation", final_eval=True),
            ]
        
        self.patch_level_tasks = [
            PatchLevelMulticlassTask(self, "cell_category", column_name="cell_category"),
        ]
        
class IMCDatasetJacksonFischer(IMCDataset):
    def __init__(self, conf, preload, filter_channel_names, filter_channel_indices):
        """
        IMCDataset class with tasks for Jackson et. al publication.
        """
        super().__init__(conf, preload, filter_channel_names, filter_channel_indices)

        self.image_level_tasks = [
            ImageLevelMulticlassTask(self, "tumor_grade", column_name="tumor_grade"),
            ImageLevelMulticlassTask(self, "tumor_type", column_name="tumor_type"),
            ImageLevelMulticlassTask(self, "tumor_clinical_type", column_name="tumor_clinical_type"),
        ]

    def setup_cell_label_encoder(self, column_name="cell_category"):
        raise NotImplemented("No cell type annotations are available for Jackson et al. dataset.")
    
    def load_cell_type_mask(self, mask_id, column_name="cell_category"):
        raise NotImplemented("No cell type annotations are available for Jackson et al. dataset.")
    
    def encode_cell_type(self, cell_type_name, column_name="cell_category"):
        raise NotImplemented("No cell type annotations are available for Jackson et al. dataset.")
    
    def decode_cell_type(self, cell_type_id, column_name="cell_category"):
        raise NotImplemented("No cell type annotations are available for Jackson et al. dataset.")
    
    def cell_type_to_id(self, column_name="cell_category"):
        raise NotImplemented("No cell type annotations are available for Jackson et al. dataset.")
    
    def id_to_cell_type(self, column_name="cell_category"):
        raise NotImplemented("No cell type annotations are available for Jackson et al. dataset.")


class IMCDatasetDanenberg(IMCDataset):
    def __init__(self, conf, preload, filter_channel_names, filter_channel_indices):
        """
        IMCDataset class with tasks for Danenberg et. al publication.
        """
        super().__init__(conf, preload, filter_channel_names, filter_channel_indices)

        self.structure_mask_buffers = dict()

        self.patch_level_tasks = [
            PatchLevelMulticlassTask(self, "predominant_cell_category", column_name="cell_category"),
        ]

        self.crop_level_tasks = [
            CropLevelBinaryStructureTask(self, "Suppressed expansion", column_name="TMEStructure-Suppressed expansion", requires_mlp=False),
            CropLevelBinaryStructureTask(self, "TLS-like", column_name="TMEStructure-TLS-like", requires_mlp=False),
            CropLevelBinaryStructureTask(self, "PDPN^{+} active stroma", column_name="TMEStructure-PDPN^{+} active stroma", requires_mlp=False),
        ]

        self.image_level_tasks = [
            ImageLevelMulticlassTask(self, "grade", column_name="Grade"),
            ImageLevelMulticlassTask(self, "PAM50", column_name="PAM50"),
            ImageLevelMulticlassTask(self, "ERStatus", column_name="ERStatus"),
        ]

    def load_structure_mask(self, mask_id, column_name):
        """
        Loads binary cell mask indicateing which cell belongs to specific structure.
        Args:
            mask_id: name of the image
            column_name: column name for single-cell annotations containing structure information.
        """
        if self.structure_mask_buffers.get(column_name) is not None:
            if self.structure_mask_buffers[column_name].get(mask_id) is not None:
                return self.structure_mask_buffers[column_name][mask_id]
        else:
            self.structure_mask_buffers[column_name] = {}

        mask = self.load_mask(mask_id)
        sc_ann = self.get_sc_annotations(mask_id)

        mapping_dict = dict(zip(sc_ann["cell_id"], sc_ann[column_name]))
        mapping_dict[0] = 0
        func = lambda x: mapping_dict.get(x, mapping_dict[0])
        mask = mask.apply_(func)
        self.structure_mask_buffers[column_name][mask_id] = mask
        return mask

class IMCDatasetCords(IMCDataset):

    def __init__(self, conf, preload, filter_channel_names, filter_channel_indices):
        """
        IMCDataset class with tasks for Cords et. al publication.
        """
        super().__init__(conf, preload, filter_channel_names, filter_channel_indices)
        self.annotations["cell_category"] = self.annotations["cell_category"].replace({"vessel": "Vessel", "T cell": "T-Cell", "Tumour": "Tumor"})
        self.annotations["cell_type"] = self.annotations["cell_type"].replace({"Bcell": "B-Cell", "normal": "Normal Tumor", "hypoxic": "Hypoxic Tumor"})

        self.patch_level_tasks = [
            PatchLevelMulticlassTask(self, "predominant_cell_category", column_name="cell_category"),
            PatchLevelMulticlassTask(self, "predominant_cell_type", column_name="cell_type"),
        ]

        self.image_level_tasks = [
            ImageLevelMulticlassTask(self, "cancer_type", column_name="cancer_type", final_eval=True),
            ImageLevelMulticlassTask(self, "lymph_node_metastatis", column_name="lymph_node_metastatis", final_eval=True),
            ImageLevelMulticlassTask(self, "grade", column_name="grade", final_eval=True),
            ImageLevelMulticlassTask(self, "Relapse", column_name="Relapse"),
        ]
