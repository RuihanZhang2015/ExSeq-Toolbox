# ExSeq-Toolbox

Welcome to ExSeq-Toolbox, a package for creating spatially-precise, three-dimensional maps of RNA localization sites. Our package has seven modules. All can be used independent of one another. 

### Modules

* **Align**: The align module is used for image alignment (registration). The chosen alignment method implemented in this module is a multiscale framework. This means that each volume pair goes through a series of downsampling steps, blurring steps, and registration steps. 

* **Args**: The args module sets up the parameters for the experiment. Specifically, the user sets the physical spacing of the experiment, the directories to save generated data, staining channel names, and the number of fields of view used. For each experiment, we instantiate an **Args** class that saves this information as attributes.

* **IO**: The io module has various functions that are used to read data, write data, visualize data, and change data types. Some of the functions include ```readXlsx```, ```readNd2```, ```tiff2H58```, ```nd2ToVol```,```nd2ToChunk``` and ```nd2ToSlice```.

* **Puncta**: The puncta module extracts puncta (concentrations of RNA molecules that can be used as a marker for RNA localization) from the image volumes, saves and concatenates their locations across staining rounds, and identifies the RNA nanoballs present. 

* **Utils**: The utils module contains functions for retrieving files. Some of the functions include ```retrieve_vol```, ```retrieve_img```, and ```retrieve_all_puncta```.

* **Segmentation**: The segmentation module includes code for running a fine-tuned Cellpose segmentation model. 

* **Stitching**: The stitching module takes independent fields of view, calculates their offset, and appropriately fuses them into one large image. For this it implements the Tile and Dataset classes, a Dataset being a list of Tiles. It also implements several loading functions and saving functions.

### Documentation
Function documentation for each of the modules can be found at [ExSeq Toolbox Documentation](https://exseq-toolbox.readthedocs.io/en/latest/). 

### Usage 
Users should clone our repository through the GitHub page and install all package dependencies via the command ```pip install -r requirements.txt```. Alternatively, users can ```pip install``` the package wheel.

### License
Distributed under the MIT License. See LICENSE.txt for more information.
