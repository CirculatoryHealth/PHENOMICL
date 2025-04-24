## PHENOMICL: a machine learning model for PHENOtyping of whole-slide images using Multiple InstanCe Learning

[![DOI](https://zenodo.org/badge/170670605.svg)](https://doi.org/10.5281/zenodo.14990008)

<!-- Please add a brief introduction to explain what the project is about    -->
Francesco Cisternino<sup>1*</sup>, Yipei Song<sup>2*</sup>, Tim S. Peters<sup>3*</sup>, Roderick Westerman<sup>3</sup>, Gert J. de Borst<sup>4</sup>, Ernest Diez Benavente<sup>3</sup>, Noortje A.M. van den Dungen<sup>3</sup>, Petra Homoed-van der Kraak<sup>5</sup>, Dominique P.V. de Kleijn<sup>4</sup>, Joost Mekke<sup>4</sup>, Michal Mokry<sup>3</sup>, Gerard Pasterkamp<sup>3</sup>, Hester M. den Ruijter<sup>3,6</sup>, Evelyn Velema<sup>6</sup>, Clint L. Miller<sup>2*</sup>, Craig A. Glastonbury<sup>1,7*</sup>, S.W. van der Laan<sup>2,3*</sup>.

<sup>* these authors contributed equally</sup>

_**Affiliations**_<br>
_<sup>1</sup> Human Technopole, Viale Rita Levi-Montalcini 1, 20157, Milan, Italy;_ 
_<sup>2</sup> Department of Genome Sciences, University of Virginia, Charlottesville, VA, USA;_ 
_<sup>3</sup> Central Diagnostic Laboratory, Division Laboratories, Pharmacy, and Biomedical genetics,  University Medical Center Utrecht, Utrecht University, Utrecht, the Netherlands;_ 
_<sup>4</sup> Vascular surgery, University Medical Center Utrecht, Utrecht University, Utrecht, the Netherlands;_ 
_<sup>5</sup> Pathology, University Medical Center Utrecht, Utrecht University, Utrecht, the Netherlands;_ 
_<sup>6</sup> Experimental Cardiology, Department Cardiology, Division Heart & Lungs, University Medical Center Utrecht, Utrecht University, Utrecht, the Netherlands;_ 
_<sup>7</sup> Nuffield Department of Medicine, University of Oxford, Oxford, UK._


### Background

Despite tremendous medical progress, cardiovascular diseases (CVD) are still topping global charts of morbidity and mortality. Atherosclerosis is the major underlying cause of CVD and results in atherosclerotic plaque formation. The extent and type of atherosclerosis is manually assessed through histological analysis, and histological characteristics are linked to major acute cardiovascular events (MACE). However, conventional means of assessing plaque characteristics suffer major limitations directly impacting their predictive power. PHENOMICL will use a machine learning method, multiple instance learning (MIL), to develop an internal representation of the 2-dimensional plaque images, allowing the model to learn position and scale in variant structures in the data.  We created a powerful model for image recognition problems using whole-slide images from stained atherosclerotic plaques to predict relevant phenotypes, for example intraplaque haemorrhage.

This work is associated with the [PHENOMICL_downstream](https://github.com/CirculatoryHealth/PHENOMICL_downstream) project.


<!-- Please add a brief introduction to explain what the project is about    -->

### Where do I start?

## Folder structure

- `scripts`: example bash (driver) scripts to run the pre-processing, training and evaluation.
- `examples`: example input files.
- `wsi_preprocessing`: pre-processing scripts.
- `/AtheroExpressCLAM/iph.py`: scripts for generating visualization of IPH heatmap

## System requirements

Software dependencies and versions are listed in requirements.txt

## Installation
First, clone this git repository: `git clone https://github.com/CirculatoryHealth/PHENOMICL.git`

Then, create and activate a conda environment: `conda env create -f phenomicl.yml` and activate: `conda activate phenomicl`

[OPTIONAL] If you have a compatible GPU and want to leverage CUDA for PyTorch, upgrade PyTorch to the GPU version:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
```

<!-- Finally, install [Openslide](https://openslide.org/download/) (>v3.1.0)

Expected installation time in normal Linux environment: 15 mins  -->

## Pre-processing - Example

Time required:
- Macbook Air M1, CPU (NO CUDA), processing 3 example WSIs: ~ 1 hour.
- Lunix server, Rocky8, GPU (CUDA), processing 3 example WSIs: ~ 1 hour.

Scripts for pre-processing are located in the `wsi_preprocessing` folder. 

### Set working directory

```bash
SLIDE_DIR="/full_path_to_where_the_wsi_are"
# Example:
# SLIDE_DIR="</full/path/to>/PHENOMICL/examples/HE/INPUT"
```

### Step 1: Segmentation & Patch extraction

```bash
python ./wsi_preprocessing/segmentation.py \
--slide_dir="${SLIDE_DIR}" \
--output_dir="${SLIDE_DIR}/PROCESSED" \
--masks_dir="${SLIDE_DIR}/PROCESSED/masks/" \
--model ./wsi_preprocessing/PathProfiler/tissue_segmentation/checkpoint_ts.pth
```

### Step 3: Feature extraction

Could throw error like: `Permission denied: '<path/to/dir>/.cache/torch'`.
In that case, please make the torch directory manually

```bash
python ./wsi_preprocessing/extract_features.py \
-h5_data="${SLIDE_DIR}/PROCESSED/patches/" \
-slide_folder="${SLIDE_DIR}" \
-output_dir="${SLIDE_DIR}/PROCESSED/features/" 
```

## Running model - Example

### Set working directory
```bash
SLIDE_DIR="/full_path_to_where_the_wsi_are"
# Example:
# SLIDE_DIR="</full/path/to>/PHENOMICL/examples/HE"
```

### Running model on pre-processed slides
```bash
python3 ./AtheroExpressCLAM/iph.py \
--h5_dir="${SLIDE_DIR}/PROCESSED/features/h5_files/" \
--csv_in="${SLIDE_DIR}/phenomicl_test_set.csv" \
--csv_out="${SLIDE_DIR}/phenomicl_test_results.csv" \
--out_dir="${SLIDE_DIR}/heatmaps/" \
--model_checkpoint="${SLIDE_DIR}/MODEL_CHECKPOINT.pt" 
```



### Project structure

<!--  You can add rows to this table, using "|" to separate columns.         -->
File                | Description                | Usage         
------------------- | -------------------------- | --------------
README.md           | Description of project     | Human editable
PHENOMICL.Rproj     | Project file               | Loads project 
LICENSE             | User permissions           | Read only     
.worcs              | WORCS metadata YAML        | Read only     
renv.lock           | Reproducible R environment | Read only     
images              | Images used in readme, etc | Human editable
scripts             | Script to process data     | Human editable

<!--  You can consider adding the following to this file:                    -->
<!--  * A citation reference for your project                                -->
<!--  * Contact information for questions/comments                           -->
<!--  * How people can offer to contribute to the project                    -->
<!--  * A contributor code of conduct, https://www.contributor-covenant.org/ -->

### Reproducibility

This project uses the Workflow for Open Reproducible Code in Science (WORCS) to
ensure transparency and reproducibility. The workflow is designed to meet the
principles of Open Science throughout a research project. 

To learn how WORCS helps researchers meet the TOP-guidelines and FAIR principles,
read the preprint at https://osf.io/zcvbs/

#### WORCS: Advice for authors

* To get started with `worcs`, see the [setup vignette](https://cjvanlissa.github.io/worcs/articles/setup.html)
* For detailed information about the steps of the WORCS workflow, see the [workflow vignette](https://cjvanlissa.github.io/worcs/articles/workflow.html)

#### WORCS: Advice for readers

Please refer to the vignette on [reproducing a WORCS project]() for step by step advice.
<!-- If your project deviates from the steps outlined in the vignette on     -->
<!-- reproducing a WORCS project, please provide your own advice for         -->
<!-- readers here.                                                           -->

### Questions and issues

<!-- Do you have burning questions or do you want to discuss usage with other users? Please use the Discussions tab.-->

Do you have burning questions or do you want to discuss usage with other users? Do you want to report an issue? Or do you have an idea for improvement or adding new features to our method and tool? Please use the [Issues tab](https://github.com/CirculatoryHealth/EntropyMasker/issues).


### Citations

Using our **`PHENOMICL`** method? Please cite our work:

    Intraplaque haemorrhage quantification and molecular characterisation using attention based multiple instance learning
    Francesco Cisternino, Yipei Song, Tim S. Peters, Roderick Westerman, Gert J. de Borst, Ernest Diez Benavente, Noortje A.M. van den Dungen, Petra Homoed-van der Kraak, Dominique P.V. de Kleijn, Joost Mekke, Michal Mokry, Gerard Pasterkamp, Hester M. den Ruijter, Evelyn Velema, Clint L. Miller, Craig A. Glastonbury, S.W. van der Laan.
    medRxiv 2025.03.04.25323316; doi: https://doi.org/10.1101/2025.03.04.25323316.


### Data availability

The whole-slide images used in this project are available through a [DataverseNL repository](https://doi.org/10.34894/QI135J "ExpressScan: Histological whole-slide image data from the Athero-Express (AE) and Aneurysm-Express (AAA) Biobank Studies"). There are restrictions on use by commercial parties, and on sharing openly based on (inter)national laws, regulations and the written informed consent. Therefore these data (and additional clinical data) are only available upon discussion and signing a Data Sharing Agreement (see Terms of Access) and within a specially designed UMC Utrecht provided environment.

### Acknowledgements

We are thankful for the support of the Netherlands CardioVascular Research Initiative of the Netherlands Heart Foundation (CVON 2011/B019 and CVON 2017-20: Generating the best evidence-based pharmaceutical targets for atherosclerosis [GENIUS I&II]), the ERA-CVD program 'druggable-MI-targets' (grant number: 01KL1802), the Leducq Fondation 'PlaqOmics' and ‘AtheroGen’, and the Chan Zuckerberg Initiative ‘MetaPlaq’. The research for this contribution was made possible by the AI for Health working group of the [EWUU alliance](https://aiforhealth.ewuu.nl/). The collaborative project ‘Getting the Perfect Image’ was co-financed through use of PPP Allowance awarded by Health~Holland, Top Sector Life Sciences & Health, to stimulate public-private partnerships.

Funding for this research was provided by National Institutes of Health (NIH) grant nos. R00HL125912 and R01HL14823 (to Clint L. Miller), a Leducq Foundation Transatlantic Network of Excellence ('PlaqOmics') grant no. 18CVD02 (to Dr. Clint L. Miller and Dr. Sander W. van der Laan), the CZI funded 'MetaPlaq' (to Dr. Clint L. Miller and Dr. Sander W. van der Laan), EU HORIZON NextGen (grant number: 101136962, to Dr. Sander W. van der Laan), EU HORIZON MIRACLE (grant number: 101115381, to Dr. Sander W. van der Laan), and Health~Holland PPP Allowance ‘Getting the Perfect Image’ (to Dr. Sander W. van der Laan).

Dr Craig A. Glastonbury has stock options in BenevolentAI and is a paid consultant for BenevolentAI, unrelated to this work. Dr. Sander W. van der Laan was funded by Roche Diagnostics, as part of 'Getting the Perfect Image', however Roche was not involved in the conception, design, execution or in any other way, shape or form of this project. 

The framework was based on the [`WORCS` package](https://osf.io/zcvbs/).

<a href='https://uefconnect.uef.fi/en/group/miracle/'><img src='images/UEF_Miracle_Logo-07.png' align="center" height="75" /></a> <a href='https://www.to-aition.eu'><img src='images/to_aition.png' align="center" height="75" /></a> <a href='https://www.health-holland.com'><img src='images/logo_NL_HealthHollland_Wit-Oranje_RGB.png' align="center" height="35" /></a> <a href='https://www.nextgentools.eu'><img src='images/NextGen_1_Red.png' align="center" height="35" /></a> <a href='https://www.era-cvd.eu'><img src='images/ERA_CVD_Logo_CMYK.png' align="center" height="75" /></a> <a href=''><img src='images/leducq-logo-large.png' align="center" height="75" /></a> <a href='https://www.fondationleducq.org'><img src='images/leducq-logo-small.png' align="center" height="75" /></a> <a href='https://osf.io/zcvbs/'><img src='images/worcs_icon.png' align="center" height="75" /></a> <a href='https://doi.org/10.1007/s10564-004-2304-6'><img src='images/AE_Genomics_2010.png' align="center" height="100" /></a>

#### Changes log

    Version:      v1.1.0
    Last update:  2025-03-04
    Written by:   Francesco Cisternino; Craig Glastonbury; Sander W. van der Laan; Clint L. Miller; Yipei Song.
    Description:  CONVOCALS repository: classification of atherosclerotic histological whole-slide images
    Minimum requirements: R version 3.4.3 (2017-06-30) -- 'Single Candle', Mac OS X El Capitan

    **MoSCoW To-Do List**
    The things we Must, Should, Could, and Would have given the time we have.
    _M_

    _S_

    _C_

    _W_
    
    Changes log
    * v1.1.0 Major overhaul, updates and re-organization prior to submission.
    * v1.0.1 Updates and re-organization.
    * v1.0.0 Initial version. 
    
    
--------------

#### Creative Commons BY-NC-ND 4.0
##### Copyright (c) 2025 [Francesco Cisternino]() \| [Craig Glastonbury](https://github.com/GlastonburyC) \| [Sander W. van der Laan](https://github.com/swvanderlaan) \| [Clint L. Miller](https://github.com/clintmil) \| [Yipei Song](https://github.com/PetraSong) 

<sup>This is a human-readable summary of (and not a substitute for) the [license](LICENSE). </sup>
</br>
<sup>You are free to share, copy and redistribute the material in any medium or format. The licencor cannot revoke these freedoms as long as you follow the license terms.</br></sup>
</br>
<sup>Under the following terms: </br></sup>
<sup><em>- Attribution</em> — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.</br></sup>
<sup><em>- NonCommercial</em> — You may not use the material for commercial purposes.</br></sup>
<sup><em>- NoDerivatives</em> — If you remix, transform, or build upon the material, you may not distribute the modified material.</br></sup>
<sup><em>- No additional</em> restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.</br></sup>
</br></sup>
<sup>Notices: </br></sup>
<sup>You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation.
No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.</sup>


