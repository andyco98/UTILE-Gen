# *UTILE-Gen* - Synthetic image generator to expand annotated datasets in the realm of nanoscience

![](https://github.com/andyco98/UTILE-Gen/blob/8dfbc183a07508691ec152c2a68591d40d733fdb/images/Bild1.png)

A preprint of the publication [UTILE-Gen: Automated Image Analysis in Nanoscience Using Synthetic Dataset Generator and Deep Learning](https://chemrxiv.org/engage/chemrxiv/article-details/6442981cdf78ec501526883f) is available on Chemrxiv for further information!

## Description
This project focus on the generation of synthetic images based on domain randomization to generate endless materials science similar datasets.
This repository contains the Python implementation of the UTILE-Gen software for the domain randomization guided creation of synthetic datasets with the corresponding instance segmentation masks.

In the following figure is possible to observe the workwise of the tool starting from one real image/mask pair, also diverse examples of how is the quality of the generated synthetic images in comparison to the original ones is depicted in the following figure:

![](https://github.com/andyco98/UTILE-Gen/blob/8dfbc183a07508691ec152c2a68591d40d733fdb/images/Bild2.png)

## Parameters
Since it is possible to fine tune a series of hyperparameters in order to modify the obtained synthetic dataset according to the individual needs, the following figure shows the different capabilities of the tool:

![](https://github.com/andyco98/UTILE-Gen/blob/8dfbc183a07508691ec152c2a68591d40d733fdb/images/Bild3.png)

## Installation
In order to run the actual version of the code, the following steps need to be done:
- Clone the repository
- Create a new environment using Anaconda using Python 3.8 or superior
- Pip install the jupyter notebook library

    ```
    pip install notebook
    ```
- From your Anaconda console open jupyter notebook (just tip "jupyter notebook" and a window will pop up)
- Open the /UTILE-Gen/UTILE-GenTool/UTILE-GenDatasetGenerator.ipynb file from the jupyter notebook directory
- Further instructions on how to use the tool are attached to the code with examples in the juypter notebook
## Dependencies
The following libraries are needed to run the program:

  ```
   pip install opencv-python
   pip install --upgrade Pillow
   pip install pyimagej
   pip install scipy
   ```
## Masks analysis
Once your model can predict on your data, you can extract the important parameters from the individual measurements using the *ROI_measurment.py*  uploaded under the *Scripts* folder.

#### Notes
- Depending on your environment additional libraries could be required.  
