# GPL 3.0 License, JSL from BIOZ
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


def get_3dzd_121_descriptor_02_pt(zmoment_scaled):
    real_part = torch.real(zmoment_scaled)
    is_nan = torch.isnan(real_part)
    zmoment_scaled = torch.where(is_nan, torch.zeros_like(zmoment_scaled), zmoment_scaled)
    
    zmoment_scaled_norm = torch.abs(zmoment_scaled) ** 2

    zmoment_scaled_norm_positive = torch.sum(zmoment_scaled_norm, dim=2)

    zero_matrix = torch.zeros_like(zmoment_scaled_norm[:, :, 0:1])
    part_matrix = zmoment_scaled_norm[:, :, 1:]
    zmoment_scaled_norm = torch.cat((zero_matrix, part_matrix), dim=2)

    zmoment_scaled_norm_negative = torch.sum(zmoment_scaled_norm, dim=2)

    zm_3dzd_invariant = torch.sqrt(zmoment_scaled_norm_positive + zmoment_scaled_norm_negative)

    zm_3dzd_invariant = torch.where(
        zm_3dzd_invariant < 1e-20, 
        torch.full_like(zm_3dzd_invariant, float('nan')), 
        zm_3dzd_invariant
    )

    return zm_3dzd_invariant