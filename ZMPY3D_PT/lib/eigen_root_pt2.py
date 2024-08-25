# MIT License
#
# Copyright (c) 2024 Jhih-Siang Lai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

def eigen_root_pt2(poly_coefficient_list):
    num_of_rows, num_of_cols = poly_coefficient_list.shape

    companion_matrix = torch.eye(num_of_cols - 2, num_of_cols - 1,
                                 dtype=torch.complex128).repeat(num_of_rows, 1, 1)

    col_1st = poly_coefficient_list[:, 0].unsqueeze(1)
    row_1st = -poly_coefficient_list[:, 1:] / col_1st

    row_1st = row_1st.unsqueeze(1)

    full_matrix = torch.cat([row_1st, companion_matrix], dim=1)
    eigenvalues = torch.linalg.eigvals(full_matrix)

    return eigenvalues
    