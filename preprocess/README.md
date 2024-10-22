<!-- In this section , it includes the notebook to precessing the dataset from downloading to homology analysis to final processed dataset (already in "data" folder)

It includes two notebooks, 1 for protein aggregation dataset, and another one for antibody aggregation dataset.

To run Aggrescan3D tool for antibody dataset, you need to have the a3d package installed , go see :  [A3D package](https://bitbucket.org/lcbio/aggrescan3d/src/master/) for each machine type (linux, window, macos)


## The environment a3d is used basically for running A3D for protein, the installation might difer from one platform to another (macos, window, linux).
## In my case , i run on linux, if there is any trouble installing with "environment_a3d.yml" , go  https://bitbucket.org/lcbio/aggrescan3d/src/master/  to see more.



## in case of error with gfortran, find the correct architecture for your machine

# conda install -c anaconda gfortran_osx-64
conda install -c anaconda gfortran_linux-64


## in case of error related to 'Modeller', you might need to add the "key" as follow:
KEY_MODELLER=MODELIRANJE conda install -c salilab modeller


## in case you use 'dynamics ' mode of aggrescan3d, there might be some error, you need to do as follows:
# #go to this
# ~/miniconda3/envs/agg3d/lib/python2.7/site-packages/CABS/pdblib.py
# #change this
# proc = Popen([self.DSSP_COMMAND, '/dev/stdin'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
# #to this
# proc = Popen(['mkdssp', '/dev/stdin'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
 -->

# Dataset Processing for Protein and Antibody Aggregation

This section includes notebooks for processing datasets, from downloading and homology analysis to preparing the final processed dataset (already available in the "data" folder). The processing steps are split into two separate notebooks:

1. **Protein Aggregation Dataset**
2. **Antibody Aggregation Dataset**

## Running Aggrescan3D for the Antibody Dataset

To process the antibody dataset using the Aggrescan3D (A3D) tool, you need to install the A3D package. Please visit the following link for installation instructions for your platform: [A3D package](https://bitbucket.org/lcbio/aggrescan3d/src/master/).

### A3D Environment Setup
The environment setup for running A3D may differ depending on the operating system (Linux, macOS, or Windows). In this project, Linux was used. If you encounter issues installing the environment using the provided `env_a3d.yml` file, refer to the A3D installation guide at the link above for more details [A3D package](https://bitbucket.org/lcbio/aggrescan3d/src/master/).

