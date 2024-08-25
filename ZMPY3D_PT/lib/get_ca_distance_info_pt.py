# GPL3.0 License, JSLai
#
# Copyright (C) 2024  Jhih-Siang Lai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch

def get_ca_distance_info_pt(xyz):
    # xyz == Nx3 torch matrix
    xyz_center = torch.mean(xyz, axis=0)
    xyz_diff = xyz - xyz_center
    xyz_dist2center = torch.sqrt(torch.sum(torch.square(xyz_diff), axis=1))

    percentiles_for_geom = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    # PyTorch doesn't support percentiles natively
    prctile_list = np.percentile(xyz_dist2center.cpu().numpy(), percentiles_for_geom)
    prctile_list = torch.tensor(prctile_list).reshape(-1, 1)

    std_xyz_dist2center = torch.std(xyz_dist2center, unbiased=False) # False to make it consistent to tf and bioz versions

    n = xyz_dist2center.shape[0]

    mean_distance = torch.mean(xyz_dist2center)
    std_xyz_dist2center = std_xyz_dist2center * torch.sqrt(torch.tensor(n / (n - 1.0)))

    s = (n / ((n - 1.0) * (n - 2.0))) * torch.sum(((xyz_dist2center - mean_distance) / std_xyz_dist2center) ** 3)

    fourth_moment = torch.sum(((xyz_dist2center - mean_distance) / std_xyz_dist2center) ** 4)

    k = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * fourth_moment -
         3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    return prctile_list, std_xyz_dist2center, s, k
