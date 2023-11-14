# Light Field Networks
### [Project Page](https://vsitzmann.github.io/lfns) | [Paper](https://arxiv.org/abs/2106.02634)  | [Data](https://drive.google.com/drive/folders/15u6WD0zSBXzu8jZBF-Sn5n01F2HSxFCp?usp=sharing) | [Pretrained Models](https://drive.google.com/drive/folders/15u6WD0zSBXzu8jZBF-Sn5n01F2HSxFCp?usp=sharing)

[Vincent Sitzmann](https://vsitzmann.github.io/)\*,
[Semon Rezchikov](https://math.columbia.edu/~skr/)\*,
[William Freeman](),
[Joshua Tenenbaum](),
[Frédo Durand]()<br>
MIT, \*denotes equal contribution

This is the official implementation of the paper "Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering".

[![lfns_video](https://img.youtube.com/vi/x3sSreTNFw4/0.jpg)](https://www.youtube.com/watch?v=x3sSreTNFw4&feature=emb_imp_woyt)

### Train Scene Representation Networks (NeurIPS 2019) with the same codebase!
In the branch "scene_rep_nets", we re-implemented our 2019 NeurIPS paper [Scene Representation Networks](https://www.vincentsitzmann.com/srns/), wich first
proposed differentiable rendering of 3D neural fields using a sphere-tracing based renderer, enabling prior-based 3D 
reconstruction from only a single image observation. You can train
both SRNs and LFNs using this codebase (all the instructions below also apply to SRNs)! Thanks to my MIT colleague 
Thomas O'Connell for helping with the SRNs re-implementation!

## Get started
You can set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate lf
```

## High-Level structure
The code is organized as follows:
* dataio.py contain the dataio for single-class experiments.
* models.py contains the code for light field networks.  
* training.py contains a generic training routine.
* ./experiment_scripts/ contains scripts to reproduce experiments in the paper.

## Reproducing experiments
The directory `experiment_scripts` contains one script per experiment in the paper.

train_single_class.py trains a model on classes in the Scene Representation Networks format, such as cars or chairs.
Note that since these datasets have a resolution of 128, this model starts with a lower resolution (64) and then 
increases the resolution to 128 (see line 43 in the script).

To reconstruct test objects, use the script "rec_single_class.py". In addition to the data root,
you have to point these scripts to the checkpoint from the training run. Note that rec_single_class.py
per default reconstructs from the view with index 64.

```
python experiment_scripts/rec_single_class.py --data_root=path_to_single_class_TEST_SET --checkpoint=path_to_training_checkpoint
```

Finally, you may test the models on the test set with the test.py script. This script is used for testing all the models. 

```
python experiment_scripts/test.py --data_root=path_to_single_class_TEST_SET --dataset=single --checkpoint=path_to_rec_checkpoint
```

To monitor progress, both the training and reconstruction scripts write tensorboard summaries into a "summaries" subdirectory in the logging_root.

## Bells & whistles
This code has a bunch of options that were not discussed in the paper.
- switch between a ReLU network and a SIREN to better fit high-frequency content with the flag --network (see the __init__ of model.py for options).
- switch between a hypernetwork, conditioning via concatenation, and low-rank concditioning with the flag --conditioning
- there is an implementation of encoder-based inference in models.py (LFEncoder) which uses a ResNet18 with global conditioning
 to generate the latent codes z.

## Data
In this implementation, unlike in the original paper, we only use one type of dataset the: class-specific ones. 

- They are loaded via dataio.py, and the paper's authors provide them in a fast-to-load hdf5 format in their 
  [google drive](https://drive.google.com/drive/folders/15u6WD0zSBXzu8jZBF-Sn5n01F2HSxFCp?usp=sharing).

### Coordinate and camera parameter conventions
This code uses an "OpenCV" style camera coordinate system, where the Y-axis points downwards (the up-vector points in the negative Y-direction), 
the X-axis points right, and the Z-axis points into the image plane. Camera poses are assumed to be in a "camera2world" format,
i.e., they denote the matrix transform that transforms camera coordinates to world coordinates.

## Misc
### Related Projects & Reading
In my [GitHub reading list on neural fields](https://github.com/vsitzmann/awesome-implicit-representations), I give an 
overview over prior work on neural fields & neural implicit representations.

In our CVPR 2018 paper [DeepVoxels](https://www.vincentsitzmann.com/deepvoxels/), we proposed a differentiable renderer
for overfitting a single voxel-grid based neural scene representations on a single scene, enabling photorealistic novel view synthesis
with high-frequency detail and view-dependent effects.
![Alt Text](https://www.vincentsitzmann.com/deepvoxels/img/fountain_stacked.mp4.gif)
![Alt Text](https://www.vincentsitzmann.com/deepvoxels/img/globe_stacked.mp4.gif)

In our NeurIPS 2019 paper [Scene Representation Networks](https://www.vincentsitzmann.com/srns/), we proposed 
to leverage a 3D MLP / neural field / neural implicit based neural scene representation that could be trained via differentiable rendering.

In our NeurIPS 2021 paper [Neural Implicit Representations with Periodic Activation Functions](https://www.vincentsitzmann.com/siren/), 
we proposed periodic nonlinearities for neural fields / neural implicit representations that can both fit high-frequency signals,
and can be used to parameterize signals with non-trivial higher-order derivatives.

### Citation
If you find our work useful in your research, please cite:
```
@inproceedings{sitzmann2021lfns,
               author = {Sitzmann, Vincent
                         and Rezchikov, Semon
                         and Freeman, William T.
                         and Tenenbaum, Joshua B.
                         and Durand, Fredo},
               title = {Light Field Networks: Neural Scene Representations
                        with Single-Evaluation Rendering},
               booktitle = {Proc. NeurIPS},
               year={2021}
            }
```

### Contact
If you have any questions, please email Vincent Sitzmann at sitzmann@mit.edu.
