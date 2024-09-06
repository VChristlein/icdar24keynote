# icdar24keynote 
# Title: Unraveling Scribal Authorship: New Frontiers in Writer Retrieval

slides and code of my icdar24 keynote

how to run the code:
```
python sift_vlad.py --labels_test <path_to>/icdar17_labels_test.txt --labels_train <path_to>icdar17_labels_train.txt -str .png -ste .jpg --to_binary --in_test <path_to>icdar17/test/binarized --in_train <path_to>icdar17/train/binarized --powernorm --rm_duplicates --tmp_folder sift_rm_dup --esvm
```

If you are using the code in your work, please kindly cite (for SIFT + E-SVM):
```
@article{Christlein17PR,
title = {Writer Identification Using GMM Supervectors and Exemplar-SVMs},
journal = {Pattern Recognition},
volume = {63},
pages = {258-267},
year = {2017},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2016.10.005},
url = {https://www.sciencedirect.com/science/article/pii/S0031320316303211},
author = {Vincent Christlein and David Bernecker and Florian Hönig and Andreas Maier and Elli Angelopoulou},
}
```
and for m-VLAD:
```
@INPROCEEDINGS{Christlein15ICDAR,
  author={Christlein, Vincent and Bernecker, David and Angelopoulou, Elli},
  booktitle={2015 13th International Conference on Document Analysis and Recognition (ICDAR)}, 
  title={Writer identification using VLAD encoded contour-Zernike moments}, 
  year={2015},
  pages={906-910},
  doi={10.1109/ICDAR.2015.7333893},
  month={Aug},
}
```
