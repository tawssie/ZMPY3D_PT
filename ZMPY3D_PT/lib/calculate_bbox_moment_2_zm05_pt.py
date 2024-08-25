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

def torch_complex_nan():
    # may be improved
    c = torch.zeros(1, dtype=torch.complex128)
    c[0] = complex(float('nan'), 0)
    return c


def calculate_bbox_moment_2_zm05_pt(max_order, gcache_complex, gcache_pqr_linear, gcache_complex_index, clm_cache3d, bbox_moment):
    max_n = max_order + 1

    bbox_moment = bbox_moment.permute(2, 1, 0).reshape(-1)
    bbox_moment = bbox_moment.to(dtype=torch.complex128)

    zm_geo = gcache_complex * bbox_moment[gcache_pqr_linear - 1]
    zm_geo_sum = torch.zeros(max_n * max_n * max_n, 1, dtype=torch.complex128)
    zm_geo_sum = zm_geo_sum.scatter_add(0, (gcache_complex_index.to(dtype=torch.int64) - 1), zm_geo)
    zm_geo_sum = torch.where(zm_geo_sum == 0, torch_complex_nan(), zm_geo_sum)

    zmoment_raw = zm_geo_sum * (3.0 / (4.0 * torch.pi))
    zmoment_raw = zmoment_raw.view(max_n, max_n, max_n).permute(2, 1, 0)

    zmoment_scaled = zmoment_raw * clm_cache3d

    return zmoment_scaled, zmoment_raw