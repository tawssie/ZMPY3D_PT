# MIT License, BIOZ
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
def get_transform_matrix_from_ab_list02_pt(A, B, center_scaled):
    a2pb2 = torch.square(A) + torch.square(B)
    a2mb2 = torch.square(A) - torch.square(B)

    m33_linear = torch.stack([
        torch.real(a2pb2),
        -torch.imag(a2mb2),
        2 * torch.imag(A * B),
        torch.imag(a2pb2),
        torch.real(a2mb2),
        -2 * torch.real(A * B),
        2 * torch.imag(A * torch.conj(B)),
        2 * torch.real(A * torch.conj(B)),
        torch.real(A * torch.conj(A)) - torch.real(B * torch.conj(B))
    ])

    m33 = m33_linear.reshape(3, 3)

    center_scaled = center_scaled.reshape(3, -1)

    m34 = torch.cat([m33, center_scaled], dim=1)

    m14 = torch.tensor([0, 0, 0, 1], dtype=torch.float64).reshape(-1, 4)

    m44 = torch.cat([m34, m14], dim=0)

    transform = torch.linalg.inv(m44)

    return transform
