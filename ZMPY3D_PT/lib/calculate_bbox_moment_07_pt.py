# GPL3.0 License, JSL from ZM, this is my originality, to calculate 3D bbox moment by tensordot
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

import torch

def calculate_bbox_moment_07_pt(voxel3d, max_order, x_sample, y_sample, z_sample):
    p, q, r = torch.meshgrid(
        torch.arange(1, max_order + 2, dtype=torch.float64),
        torch.arange(1, max_order + 2, dtype=torch.float64),
        torch.arange(1, max_order + 2, dtype=torch.float64),
        indexing='ij'
    )

    extend_voxel3d = torch.nn.functional.pad(voxel3d, (0, 1, 0, 1, 0, 1))
    diff_extend_voxel3d = torch.diff(torch.diff(torch.diff(extend_voxel3d, dim=0), dim=1), dim=2)

    x_sample = x_sample[1:, None]
    x_power = torch.pow(x_sample, torch.arange(1, max_order + 2, dtype=torch.float64))

    y_sample = y_sample[1:, None]
    y_power = torch.pow(y_sample, torch.arange(1, max_order + 2, dtype=torch.float64))

    z_sample = z_sample[1:, None]
    z_power = torch.pow(z_sample, torch.arange(1, max_order + 2, dtype=torch.float64))

    temp1 = torch.tensordot(x_power, diff_extend_voxel3d, dims=([0], [0]))
    temp2 = torch.tensordot(y_power, temp1, dims=([0], [1]))
    bbox_moment = torch.tensordot(z_power, temp2, dims=([0], [2]))

    bbox_moment = -(bbox_moment.permute(2, 1, 0) / (p * q * r))

    volume_mass = bbox_moment[0, 0, 0]
    center = [
        bbox_moment[1, 0, 0] / volume_mass,
        bbox_moment[0, 1, 0] / volume_mass,
        bbox_moment[0, 0, 1] / volume_mass
    ]

    center = torch.stack(center)

    
    return volume_mass, center, bbox_moment
