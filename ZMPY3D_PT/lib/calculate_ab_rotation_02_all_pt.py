# GPL3.0, JSL, ZM, bioz
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

from .eigen_root_pt import *
from .eigen_root_pt2 import *

def calculate_ab_rotation_02_all_pt(z_moment_raw, target_order_2_norm_rotate):
    if target_order_2_norm_rotate % 2 == 0:
        abconj_coef = torch.stack([
            z_moment_raw[target_order_2_norm_rotate, 2, 2],
            -z_moment_raw[target_order_2_norm_rotate, 2, 1],
            z_moment_raw[target_order_2_norm_rotate, 2, 0],
            torch.conj(z_moment_raw[target_order_2_norm_rotate, 2, 1]),
            torch.conj(z_moment_raw[target_order_2_norm_rotate, 2, 2])
        ])
        n_abconj = 4
    else:
        abconj_coef = torch.stack([
            z_moment_raw[target_order_2_norm_rotate, 1, 1],
            -z_moment_raw[target_order_2_norm_rotate, 1, 0],
            -torch.conj(z_moment_raw[target_order_2_norm_rotate, 1, 1])
        ])
        n_abconj = 2

    abconj_coef = abconj_coef.unsqueeze(0)
    abconj_sol = eigen_root_pt(abconj_coef)

    def get_ab_list_by_ind_real(ind_real):
        k_re = abconj_sol.real
        k_im = abconj_sol.imag
        k_im2 = k_im ** 2
        k_re2 = k_re ** 2
        k_im3 = k_im * k_im2
        k_im4 = k_im2 ** 2
        k_re4 = k_re2 ** 2

        f20 = z_moment_raw[ind_real, 2, 0].real
        f21 = z_moment_raw[ind_real, 2, 1]
        f22 = z_moment_raw[ind_real, 2, 2]

        f21_im = f21.imag
        f21_re = f21.real
        f22_im = f22.imag
        f22_re = f22.real

        coef4 = (
            4 * f22_re * k_im * (-1 + k_im2 - 3 * k_re2) -
            4 * f22_im * k_re * (1 - 3 * k_im2 + k_re2) -
            2 * f21_re * k_im * k_re * (-3 + k_im2 + k_re2) +
            2 * f20 * k_im * (-1 + k_im2 + k_re2) +
            f21_im * (1 - 6 * k_im2 + k_im4 - k_re4)
        )

        coef3 = (
            2 * (-4 * f22_im * (k_im + k_im3 - 3 * k_im * k_re2) +
            f21_re * (-1 + k_im4 + 6 * k_re2 - k_re4) +
            2 * k_re * (f22_re * (2 + 6 * k_im2 - 2 * k_re2) +
            f21_im * k_im * (-3 + k_im2 + k_re2) +
            f20 * (-1 + k_im2 + k_re2)))
        )

        bimbre_coef = torch.stack([coef4, coef3, torch.zeros_like(coef4), coef3, -coef4]).T

        bimbre_sol_real = eigen_root_pt2(bimbre_coef.to(torch.complex128)).real

        is_abs_bimre_good = torch.abs(bimbre_sol_real) > 1e-7

        bre = 1 / torch.sqrt((1 + (bimbre_sol_real ** 2)) * (1 + k_im2 + k_re2).unsqueeze(1))
        bim = bimbre_sol_real * bre

        b = torch.complex(bre, bim)
        a = torch.conj(b) * abconj_sol.unsqueeze(1)

        ab_list = torch.stack([a[is_abs_bimre_good], b[is_abs_bimre_good]], dim=1)

        return ab_list

    ind_real_all = torch.arange(2, z_moment_raw.size(0) + 1, step=2)

    ab_list_all = torch.stack([get_ab_list_by_ind_real(ind) for ind in ind_real_all])

    return ab_list_all

