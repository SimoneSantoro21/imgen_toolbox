import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import re

class Dataset3D(Dataset):
    """
    Unified dataset over ALL center indices in one folder.

    Expects two subfolders:
      - CENTERS:   center_{patientid}_{center_idx}.nii.gz
      - NEIGHBORS: neighbor_{patientid}_{center_idx}.nii.gz

    Returns:
      (neighbor_tensor, center_tensor)
        neighbor_tensor: [C, D, H, W]  (C can be 1 or >1)
        center_tensor:   [1, D, H, W]  (or [C, D, H, W] if your centers are multi-channel too)
    """

    def __init__(self, root_path: str, transform=None):
        self.centers_dir = os.path.join(root_path, "CENTERS")
        self.neighbors_dir = os.path.join(root_path, "NEIGHBORS")
        self.transform = transform

        pat_center   = re.compile(r'^center_([^_]+)_(\d{1,2})\.nii\.gz$')
        pat_neighbor = re.compile(r'^neighbor_([^_]+)_(\d{1,2})\.nii\.gz$')

        center_map = {}
        neighbor_map = {}

        for fname in os.listdir(self.centers_dir):
            m = pat_center.match(fname)
            if not m:
                continue
            pid, ci = m.group(1), int(m.group(2))
            center_map[(ci, pid)] = os.path.join(self.centers_dir, fname)

        for fname in os.listdir(self.neighbors_dir):
            m = pat_neighbor.match(fname)
            if not m:
                continue
            pid, ci = m.group(1), int(m.group(2))
            neighbor_map[(ci, pid)] = os.path.join(self.neighbors_dir, fname)

        self.samples = []
        for key, cpath in center_map.items():
            if key in neighbor_map:
                npath = neighbor_map[key]
                center_idx, patient_id = key
                self.samples.append((center_idx, patient_id, cpath, npath))

        self.samples.sort(key=lambda x: (x[0], x[1]))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_nii(path: str) -> np.ndarray:
        arr = nib.load(path).get_fdata().astype(np.float32)
        return arr

    @staticmethod
    def _to_torch_volume(arr: np.ndarray) -> torch.Tensor:
        """
        Accepts:
          - 3D: [H, W, D] (nibabel typical)
          - 4D: [H, W, D, C] (channels-last, e.g. axis=-1 stacking)

        Returns:
          - 3D -> [1, D, H, W]
          - 4D -> [C, D, H, W]
        """
        t = torch.from_numpy(arr)

        if t.ndim == 3:
            # [H, W, D] -> [D, H, W] -> [1, D, H, W]
            t = t.permute(2, 0, 1).contiguous()
            t = t.unsqueeze(0)
            return t

        if t.ndim == 4:
            # [H, W, D, C] -> [C, D, H, W]
            t = t.permute(3, 2, 0, 1).contiguous()
            return t

        raise ValueError(f"Expected 3D or 4D volume, got shape {tuple(t.shape)}")

    def __getitem__(self, idx):
        center_idx, patient_id, cpath, npath = self.samples[idx]

        center_arr = self._load_nii(cpath)
        neighbor_arr = self._load_nii(npath)

        center_tensor = self._to_torch_volume(center_arr)
        neighbor_tensor = self._to_torch_volume(neighbor_arr)

        return neighbor_tensor, center_tensor
