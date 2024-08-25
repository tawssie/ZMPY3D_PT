# This notebook is used for developing the CLI.
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

import numpy as np
import torch
import pickle
import argparse
import os
import sys

import ZMPY3D_PT as z

# Full procedure to calculate superposition for a single graph component.
def core(Voxel3D,Corner,GridWidth,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear,MaxOrder,s_id,n,l,m,mu,k,IsNLM_Value):

    Dimension_BBox_scaled=Voxel3D.shape
    X_sample = torch.arange(0, Dimension_BBox_scaled[0] + 1, dtype=torch.float64)
    Y_sample = torch.arange(0, Dimension_BBox_scaled[1] + 1, dtype=torch.float64)
    Z_sample = torch.arange(0, Dimension_BBox_scaled[2] + 1, dtype=torch.float64)
    
    [VolumeMass,Center,_]=z.calculate_bbox_moment(Voxel3D,1,X_sample,Y_sample,Z_sample)

    [AverageVoxelDist2Center,MaxVoxelDist2Center]=z.calculate_molecular_radius(Voxel3D,Center,VolumeMass,1.80) # Param['default_radius_multiplier'] == 1.80

    Center_scaled=Center*GridWidth+Corner
    
    ##################################################################################
    # You may add any preprocessing on the voxel before applying the Zernike moment. #
    ##################################################################################
                
    Sphere_X_sample, Sphere_Y_sample, Sphere_Z_sample=z.get_bbox_moment_xyz_sample(Center,AverageVoxelDist2Center,Dimension_BBox_scaled)
    
    _,_,SphereBBoxMoment=z.calculate_bbox_moment(Voxel3D
                                      ,MaxOrder
                                      ,Sphere_X_sample
                                      ,Sphere_Y_sample
                                      ,Sphere_Z_sample)
    
    ZMoment_scaled,ZMoment_raw=z.calculate_bbox_moment_2_zm(MaxOrder
                                       , GCache_complex
                                       , GCache_pqr_linear
                                       , GCache_complex_index
                                       , CLMCache3D
                                       , SphereBBoxMoment)

    ABList_2=z.calculate_ab_rotation_all(ZMoment_raw, 2)
    ABList_3=z.calculate_ab_rotation_all(ZMoment_raw, 3)
    ABList_4=z.calculate_ab_rotation_all(ZMoment_raw, 4)
    ABList_5=z.calculate_ab_rotation_all(ZMoment_raw, 5)
    ABList_6=z.calculate_ab_rotation_all(ZMoment_raw, 6)

    ABList_all = torch.cat([torch.reshape(ABList_2, (-1, 2)),torch.reshape(ABList_3, (-1, 2)),torch.reshape(ABList_4, (-1, 2)),torch.reshape(ABList_5, (-1, 2)),torch.reshape(ABList_6, (-1, 2))], dim=0)
    ZMList_all = z.calculate_zm_by_ab_rotation(ZMoment_raw, BinomialCache, ABList_all, MaxOrder, CLMCache,s_id,n,l,m,mu,k,IsNLM_Value)
    ZMList_all = ZMList_all[~torch.isnan(ZMList_all.real)]
    ZMList_all = torch.reshape(ZMList_all, (-1, 96))
    
    return Center_scaled, ABList_all,ZMList_all


def CalMatrix(data1, data2):
    Center_scaled_A=data1[0]
    ABList_A=data1[1]
    ZMList_A=data1[2]

    Center_scaled_B=data2[0]
    ABList_B=data2[1]
    ZMList_B=data2[2]


    M = torch.abs(torch.matmul(torch.conj(ZMList_A).t(), ZMList_B))    
    MaxValueIndex = torch.nonzero(M == torch.max(M))
        
    i,j=MaxValueIndex[0,0], MaxValueIndex[0,1]
    
    RotM_A=z.get_transform_matrix_from_ab_list(ABList_A[i,0],ABList_A[i,1],Center_scaled_A)
    RotM_B=z.get_transform_matrix_from_ab_list(ABList_B[j,0],ABList_B[j,1],Center_scaled_B)
    
    TargetRotM = torch.linalg.solve(RotM_B, RotM_A)

    return TargetRotM


def ZMPY3D_PT_CLI_BatchSuperA2B(PDBFileNameA, PDBFileNameB):
    
    MaxOrder=int(6)
    GridWidth= 1.00

    BinomialCacheFilePath = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_data'), 'BinomialCache.pkl')
    with open(BinomialCacheFilePath, 'rb') as file: # Used at the entry point, it requires __file__ to identify the package location
    # with open('./cache_data/BinomialCache.pkl', 'rb') as file: # Can be used in ipynb, but not at the entry point. 
        BinomialCachePKL = pickle.load(file)

    LogCacheFilePath=os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_data'), 'LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder))
    with open(LogCacheFilePath, 'rb') as file: # Used at the entry point, it requires __file__ to identify the package location
    # with open('./cache_data/LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder), 'rb') as file: # Can be used in ipynb, but not at the entry point. 
        CachePKL = pickle.load(file)  

    # Extract all cached variables from pickle. These will be converted into a tensor/cupy objects for ZMPY3D_CP, ZMPY3D_TF and ZMPY3D_PT.
    BinomialCache = torch.tensor(BinomialCachePKL['BinomialCache'], dtype=torch.float64)

    # GCache, CLMCache, and all RotationIndex
    GCache_pqr_linear = torch.tensor(CachePKL['GCache_pqr_linear'])
    GCache_complex = torch.tensor(CachePKL['GCache_complex'])
    GCache_complex_index = torch.tensor(CachePKL['GCache_complex_index'])
    CLMCache3D = torch.tensor(CachePKL['CLMCache3D'], dtype=torch.complex128)
    CLMCache = torch.tensor(CachePKL['CLMCache'], dtype=torch.float64)

    RotationIndex = CachePKL['RotationIndex']
    
    # RotationIndex is a structure, must be [0,0] to accurately obtain the s_id ... etc, within RotationIndex.
    s_id = torch.tensor(np.squeeze(RotationIndex['s_id'][0,0]) - 1, dtype=torch.int64)
    n    = torch.tensor(np.squeeze(RotationIndex['n'][0,0]), dtype=torch.int64)
    l    = torch.tensor(np.squeeze(RotationIndex['l'][0,0]), dtype=torch.int64)
    m    = torch.tensor(np.squeeze(RotationIndex['m'][0,0]), dtype=torch.int64)
    mu   = torch.tensor(np.squeeze(RotationIndex['mu'][0,0]), dtype=torch.int64)
    k    = torch.tensor(np.squeeze(RotationIndex['k'][0,0]), dtype=torch.int64)
    IsNLM_Value = torch.tensor(np.squeeze(RotationIndex['IsNLM_Value'][0,0]) - 1, dtype=torch.int64)



    Param=z.get_global_parameter()
    ResidueBox=z.get_residue_gaussian_density_cache(Param)


    Center_scaled_A=[]
    ABList_all_A=[]
    ZMList_all_A=[]
    
    for f in PDBFileNameA:
        [XYZ,AA_NameList]=z.get_pdb_xyz_ca(f)

        [Voxel3D,Corner]=z.fill_voxel_by_weight_density(XYZ,AA_NameList,Param['residue_weight_map'],GridWidth,ResidueBox[GridWidth])

        Voxel3D=torch.tensor(Voxel3D,dtype=torch.float64)
        Corner =torch.tensor(Corner ,dtype=torch.float64)
        

        total_residue_weight = z.get_total_residue_weight(AA_NameList,Param['residue_weight_map'])
        total_residue_weight_tensor = torch.tensor(total_residue_weight, dtype=torch.float64)
        
        xyz_tensor = torch.tensor(XYZ, dtype=torch.float64)


        Center_scaled, ABList_all,ZMList_all =core(Voxel3D, Corner, GridWidth, 
                   BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, 
                   GCache_pqr_linear, MaxOrder, s_id, n, l, m, mu, k, IsNLM_Value)

        Center_scaled_A.append(Center_scaled)
        ABList_all_A.append(ABList_all)
        ZMList_all_A.append(ZMList_all)



    Center_scaled_B=[]
    ABList_all_B=[]
    ZMList_all_B=[]
    
    for f in PDBFileNameB:
        [XYZ,AA_NameList]=z.get_pdb_xyz_ca(f)

        [Voxel3D,Corner]=z.fill_voxel_by_weight_density(XYZ,AA_NameList,Param['residue_weight_map'],GridWidth,ResidueBox[GridWidth])

        Voxel3D=torch.tensor(Voxel3D,dtype=torch.float64)
        Corner =torch.tensor(Corner ,dtype=torch.float64)
        

        total_residue_weight = z.get_total_residue_weight(AA_NameList,Param['residue_weight_map'])
        total_residue_weight_tensor = torch.tensor(total_residue_weight, dtype=torch.float64)
        
        xyz_tensor = torch.tensor(XYZ, dtype=torch.float64)


        Center_scaled, ABList_all,ZMList_all =core(Voxel3D, Corner, GridWidth, 
                   BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, 
                   GCache_pqr_linear, MaxOrder, s_id, n, l, m, mu, k, IsNLM_Value)

        Center_scaled_B.append(Center_scaled)
        ABList_all_B.append(ABList_all)
        ZMList_all_B.append(ZMList_all)

    
    TargetRotMList=[]

    for center_a, ab_a, zm_a, center_b, ab_b, zm_b in zip(Center_scaled_A,ABList_all_A,ZMList_all_A,Center_scaled_B,ABList_all_B,ZMList_all_B):
        data1 = (center_a, ab_a, zm_a)
        data2 = (center_b, ab_b, zm_b)
        
        # Call the function with the prepared tuples
        TargetRotM = CalMatrix(data1, data2)
        TargetRotMList.append(TargetRotM)

    return TargetRotMList

def main():
    if len(sys.argv) != 2:
        print('Usage: ZMPY3D_PT_CLI_BatchSuperA2B PDBFileList.txt')
        print('       This function takes a list of paired PDB structure file paths to generate transformation matrices.')
        print("Error: You must provide exactly one input file.")
        sys.exit(1)

    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        print("CUDA is available, PyTorch uses GPU.")
    else:
        print("CUDA is unavailable. PyTorch uses CPU.")


    parser = argparse.ArgumentParser(description='Process input file that contains paths to .pdb or .txt files.')
    parser.add_argument('input_file', type=str, help='The input file that contains paths to .pdb or .txt files.')

    args = parser.parse_args()

    # Perform validation checks directly after parsing arguments
    input_file = args.input_file
    if not input_file.endswith('.txt'):
        parser.error("File must end with .txt")
    
    if not os.path.isfile(input_file):
        parser.error("File does not exist")

    with open(input_file, 'r') as file:
        lines = file.readlines()

    file_list_1 = []
    file_list_2 = []
    for line in lines:

        files = line.strip().split()
        if len(files) != 2:
            print(f"Error: Each line must contain exactly two file paths, but got {len(files)}.")
            sys.exit(1)
        file1, file2 = files
        
        for file in [file1, file2]:
            if not (file.endswith('.pdb') or file.endswith('.txt')):
                print(f"Error: File {file} must end with .pdb or .txt.")
                sys.exit(1)
            if not os.path.isfile(file):
                print(f"Error: File {file} does not exist.")
                sys.exit(1)
        file_list_1.append(file1)
        file_list_2.append(file2)

    TargetRotM=ZMPY3D_PT_CLI_BatchSuperA2B(file_list_1, file_list_2)

    for M in TargetRotM:
        print(M)

if __name__ == '__main__':
    main()