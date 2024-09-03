# icdar24keynote 
# Title: Unraveling Scribal Authorship: New Frontiers in Writer Retrieval

slides and code of my icdar24 keynote

how to run the code:
```
python sift_vlad.py --labels_test /home/christlein_local/teaching/projcv/writerident/exercise/data/icdar17_labels_test.txt --labels_train /home/christlein_local/teaching/projcv/writerident/exercise/data/icdar17_labels_train.txt -str .png -ste .jpg --to_binary --in_test /run/media/christlein_local/Data/icdar17/test/binarized --in_train /run/media/christlein_local/Data/icdar17/train/binarized --powernorm --rm_duplicates --load_folder sift_rm_dup --tmp_folder sift_rm_dup --esvm
```
