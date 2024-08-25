# GPL3.0 License, JSL, ZM
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


def calculate_zm_by_ab_rotation01_pt(z_moment_raw, binomial_cache, ab_list, max_order, clm_cache, s_id, n, l, m, mu, k, is_nlm_value):
    a = ab_list[:, 0]
    b = ab_list[:, 1]

    aac = (a * torch.conj(a)).real.to(dtype=torch.complex128)
    bbc = (b * torch.conj(b)).real.to(dtype=torch.complex128)
    bbcaac = -bbc / aac
    abc = -(a / torch.conj(b))
    ab = a / b


    bbcaac_pow_k_list = torch.log(bbcaac)[:, None] * torch.arange(max_order + 1, dtype=torch.float32).to(torch.complex128)
    aac_pow_l_list = torch.log(aac)[:, None] * torch.arange(max_order + 1, dtype=torch.float32).to(torch.complex128)
    ab_pow_m_list = torch.log(ab)[:, None] * torch.arange(max_order + 1, dtype=torch.float32).to(torch.complex128)
    abc_pow_mu_list = torch.log(abc)[:, None] * torch.arange(-max_order, max_order + 1, dtype=torch.float32).to(torch.complex128)
    
    f_exp = torch.zeros_like(s_id, dtype=torch.complex128)

    cond1 = mu >= 0
    cond2 = (mu < 0) & (mu % 2 == 0)
    cond3 = (mu < 0) & (mu % 2 != 0)


    f_exp_values1 = z_moment_raw[ n[cond1], l[cond1], mu[cond1]  ]
    f_exp_values2 = torch.conj(z_moment_raw[ n[cond2], l[cond2], -mu[cond2]  ])
    f_exp_values3 = -torch.conj(z_moment_raw[n[cond3], l[cond3], -mu[cond3]])
    
    def update_by_cond(tensor, cond, values):
        tensor[cond] = values
        return tensor
    
    f_exp = update_by_cond(f_exp, cond1, f_exp_values1)
    f_exp = update_by_cond(f_exp, cond2, f_exp_values2)
    f_exp = update_by_cond(f_exp, cond3, f_exp_values3)
    
    f_exp = torch.log(f_exp)

    max_n = max_order + 1
    clm = clm_cache[l * max_n + m]
    clm = clm.to(dtype=torch.complex128)
    clm = clm.squeeze()

    indices_bin1 = torch.stack([l - mu, k - mu], dim=1)
    indices_bin2 = torch.stack([l + mu, k - m], dim=1)
    
    bin_part1 = binomial_cache[indices_bin1[:, 0], indices_bin1[:, 1]]
    bin_part2 = binomial_cache[indices_bin2[:, 0], indices_bin2[:, 1]]
    
    bin = bin_part1.to(dtype=torch.complex128) + bin_part2.to(dtype=torch.complex128)

    al = aac_pow_l_list[:, l]
    abpm = ab_pow_m_list[:, m]
    amu = abc_pow_mu_list[:, max_order + mu]
    bbk = bbcaac_pow_k_list[:, k]
    
    nlm = f_exp + clm + bin + al + abpm + amu + bbk

    exp_nlm = torch.exp(nlm)
    exp_nlm = exp_nlm.T
    
    z_nlm = torch.zeros_like(is_nlm_value, dtype=torch.complex128)
    z_nlm = z_nlm.unsqueeze(1)
    z_nlm = z_nlm.repeat(1, a.numel())
    
    z_nlm = torch.index_add(z_nlm, dim=0, index=s_id, source=exp_nlm)


    zm = torch.full((z_moment_raw.numel(), ab_list.size(0)), float('nan'), dtype=torch.complex128)

    zm[is_nlm_value, :] = z_nlm
    
    zm = zm.view(max_n, max_n, max_n, -1)
    zm = zm.permute(2, 1, 0, 3)

    return zm
