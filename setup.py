from setuptools import setup, find_packages

setup(
    name='ZMPY3D_PT',
    version='0.0.3',
    author='Jhih Siang (Sean) Lai',
    author_email='js.lai@uqconnect.edu.au, jsl035@ucsd.edu',
    description='ZMPY3D PyTorch version',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tawssie/ZMPY3D_PT',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],

    entry_points={
        'console_scripts': [
            'ZMPY3D_PT_CLI_ZM=ZMPY3D_PT.ZMPY3D_PT_CLI_ZM:main',
            'ZMPY3D_PT_CLI_SuperA2B=ZMPY3D_PT.ZMPY3D_PT_CLI_SuperA2B:main',
            'ZMPY3D_PT_CLI_ShapeScore=ZMPY3D_PT.ZMPY3D_PT_CLI_ShapeScore:main',
            'ZMPY3D_PT_CLI_BatchSuperA2B=ZMPY3D_PT.ZMPY3D_PT_CLI_BatchSuperA2B:main',
            'ZMPY3D_PT_CLI_BatchShapeScore=ZMPY3D_PT.ZMPY3D_PT_CLI_BatchShapeScore:main',
            'ZMPY3D_PT_CLI_BatchZM=ZMPY3D_PT.ZMPY3D_PT_CLI_BatchZM:main',
        ],
    },

    python_requires='>=3.9.16',
    install_requires=[
        'numpy>=1.23.5',
        # 'torch>=2.3.1' # remove the requirement to avoid conflicts between cuda11x,12x
    ],

    include_package_data=True, # for the cache
)


