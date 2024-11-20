<div align=center>
  <h1>
  🪑 3D Volume Generation
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs492d-fall-2024/ target="_blank"><b>KAIST CS492(D): Diffusion Models and Their Applications (Fall 2024)</b></a><br>
    Course Project
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://63days.github.io/ target="_blank"><b>Juil Koo</b></a>  (63days [at] kaist.ac.kr)
  </p>
</div>

<div align=center>
   <img src="./assets/teaser.png">
   <figcaption>
	  3D shapes from the <a href=https://shapenet.org/ target="_blank">ShapeNet</a> dataset. The  <a href=https://shapenet.org/ target="_blank">ShapeNet</a> dataset is a large collection of 3D models spanning various object categories. In this project, we specifically focus on three object categories: chairs, airplanes, and tables, which are among the most popular and vary in style and complexity. These 3D shapes are typically represented as 3D meshes but can be converted into various different formats: voxel grids or point clouds. We aim to build a diffusion model sampling 3D shapes represented by voxels.
    </figcaption>
</div>

## Description
This project aims to build a 3D diffusion model, specifically targeting 3D volume diffusion. We train the model using voxel data at a resolution of (128, 128, 128) from the <a href=https://shapenet.org/ target="_blank">ShapeNet</a> dataset. A major challenge will be efficiently handling this high-resolution data within limited VRAM constraints.


## Data Specification
The dataset consists of three categories of 3D shapes from the ShapeNet dataset: airplanes, chairs, and tables, with 2,658, 1,958, and 3,835 shapes for training in each category, respectively. Each shape is represented by a binary voxel, where a value of 1 indicates the object's surface at that position. To obtain the voxel data, we voxelize point clouds from the ShapeNet dataset. Run the following commands to install required dependencies and to preprocess the data:

```
pip install -r requirements.txt
python load_data.py
```

__If the ShapeNet download link is not accessible or you encounter an error running `load_data.py`, please download the dataset on your local machine from [here](https://onedrive.live.com/?authkey=%21AFhw6N804-SPITU&id=0CE615B143FC4BDC%21188222&cid=0CE615B143FC4BDC) and send the file to your remote server. You have two options: either download `hdf5_data.zip`, the original ShapeNet dataset, and run `load_data.py` for pre-processing, or directly download `.npy` files. Note that downloading the `.npy` files may take longer.__ 

To send the file in your local machine to the remote server, you can refer to the following `rsync` command:
```
rsync -azP -e "ssh -i {PATH/TO/PRIVATE_KEY}" {PATH/TO/LOCAL_FILE} root@{KCLOUD_IP}:{PATH/TO/REMOTE/DESTINATION}
```

A 3D voxel visualization code is in `visualize.ipynb`.

## Tasks
Your task is to implement diffusion models that sample 3D voxels of the three categories. You can implement either a single class-conditioned model that can sample all categories or a separate unconditional model for each category. You can even convert the provided voxel data into any format, such as meshes or point clouds so that the network takes the converted 3D data as input. 

__The only implementation requirement is that the final output of the model must be 3D voxels, i.e., the output of your network should be voxels only, not other representations, such as point clouds or meshes, that require post-processing to be converted into voxels.__

After implementing the model, run the evaluaiton code provided and report the results. Below are further details on the evaluation.


## Evaluation
To assess the quality and diversity of the generated samples, we follow Achlioptas et al. [1] and use Jensen-Shannon Divergence (JSD), Minimum Matching Distance (MMD), and Coverage (COV). 

JSD treats voxel sets as probability distributions over the 3D space, where each voxel grid represents the likelihood of being occupied. It then measures the Jensen-Shannon Divergence between the two input voxel sets, providing a similarity measure from the probabilistic perspective. In MMD and COV, we first compute the nearest neighbor in the reference set for each voxel in the sample set. COV evaluates the diversity of the samples by measuring how many voxels in the reference set are covered by the nearest neighbors from the sample set. On the other hand, MMD assesses the fidelity of the samples by calculating the distance between each voxel in the sample set and its corresponding nearest neighbor in the reference set.

__*For each category*__, sample 1,000 voxels using your model and save them in `.npy` format with a shape of `(1000, 128, 128, 128)`. At the end of the sampling process, discretize the values to either 0 or 1 by applying a threshold, setting the value to 1 if x > 0.5 and to 0 otherwise. Once the data is saved, run the following command to measure JSD, MMD, and COV:

```
python eval.py {CATEGORY} {PATH/TO/YOUR_SAMPLE_DATA.NPY}
```
, where `CATEGORY` is one of `chair`, `airplane`, or `table`.

Note that the evaluation script may take around 30 minutes to complete.


## What to Submit
In a single PDF file, report screenshots of your __quantitative scores (JSD, MMD, and COV)__ along with at least 8 visualization of your samples __for each category__.
Compress your source code and the pdf file into a zip file and submit it.

## Acknowledgement 
The dataset is from <a href=https://shapenet.org/ target="_blank">ShapeNet</a>.

## Reference
[1] [Learning Representations and Generative Models for 3D Point Clouds](https://arxiv.org/abs/1707.02392)
