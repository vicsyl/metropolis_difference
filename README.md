# Differences in objects world coordinates as provided in the [Mapillary Metropolis](https://www.mapillary.com/dataset/metropolis) dataset


### File layout

{object-type}/{distance of the two measurements}_{0 or 1}_{object_id}.png, e.g.:

[./object--support--pole/1.0234649177693373_1_gEsqliSybkEbUB7JlnCdcmZriHw3GeRzvNkL4xnQ50d8.png](./object--support--pole/1.0234649177693373_1_gEsqliSybkEbUB7JlnCdcmZriHw3GeRzvNkL4xnQ50d8.png)


### What is shown

For each object (from a subset of objects in the dataset), 
the two images are shown between which the object detected 3D bounding boxes have the largest distance.
The 3D bounding box related to the image in which it is shown is in blue, the 3D bounding box belonging 
to the other image is in red.

### Data structure

instance_id
* id of the object 
* called "instance_token" from sample_annotation.json or sample_annotation_2d.json in the dataset

sample data
* id of the image
* called (sample data) "token" in sample_data.json in the dataset

token
* id of the 3D bounding box
* called "token" in sample_annotation.json in the dataset
