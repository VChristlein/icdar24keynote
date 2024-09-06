# icdar24keynote 
# Title: Unraveling Scribal Authorship: New Frontiers in Writer Retrieval

slides and code of my icdar24 keynote

how to run the code:
```
python sift_vlad.py --labels_test <path_to>/icdar17_labels_test.txt --labels_train <path_to>icdar17_labels_train.txt -str .png -ste .jpg --to_binary --in_test <path_to>icdar17/test/binarized --in_train <path_to>icdar17/train/binarized --powernorm --rm_duplicates --tmp_folder sift_rm_dup --esvm
```
