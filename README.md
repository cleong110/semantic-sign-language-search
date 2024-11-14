# Semantic Sign Language Video Search.
Search a folder of sign language videos semantically. Given a video, find other videos which are semantically related, using [SignCLIP Embeddings](https://github.com/J22Melody/fairseq/tree/main/examples/MMPT). 

Currently very WIP and the code is a bit rough, but I got it working on ASL Citizen. The code needs reworking to, for example, simplify the setup process, rework the database schema to put "dataset" into its own table, and so forth. I have a half-finished branch which got a streamlit demo running.

## Overview
Basic process is as follows.
1. Take your folder of videos and preprocess them to .pose files with `pose-format`. See the section on parallelizing below.
2. (not implemented) segment the videos into individual signs. I didn't implement this yet, because ASL Citizen is already segmented, and I had technical trouble with the segmentation software.
3. Take your .pose files and embed them with SignCLIP, saving as .npy. I have notes on this in the setup_signCLIP folder
4. Take your .npy files and load them into a pgvector database. Then you can do efficient queries, ordering by embedding vector distances. Notes on this are in the setup_db folder.

## Experiments on ASL Citizen with 4 models. 

### SignCLIP Models
Zifan Jiang (author of SignCLIP) gave me access to four models: 
1. baseline pretrained on SpreadTheSign,
2. finetuned on asl_signs,
3. finetuned on asl_citizen,
4. finetuned on sem_lex

### Subsets
From ASL Citizen I sampled some videos for initial testing:
* “20x5”, manually curated videos with glosses that seemed semantically related. 5 videos randomly selected from the full set, for each sign.
* “400x10”, 400 randomly-chosen signs, with 10 random videos chosen from the full set per sign.
* Test Set: 33k videos, as described in the ASL Citizen splits 
* Full set: all videos, about 83k videos
Note: Results for the the model finetuned on asl_citizen can only be considered valid for the test set!

### All against All Retrieval
I ran "all against all experiments thus:
```
For each subset
	For each model
		For each video 
		Retrieve top 10 most similar by L2 Distance using pgvector
		If gloss equals, “match”!
```

The folder of results can be found [here](https://drive.google.com/drive/folders/15ZrzPb1SaryDKXTEgohuWbS9yENaUF4o?usp=drive_link), including raw outputs and various analysis files.

### "Most confused" analysis
I parsed the output logs to calculate which glosses were most-confused with which other glosses. 

![image](https://drive.google.com/uc?export=view&id=1iZY3_CcpFPeA57j03YQy9ApQ7A5JxqTp)
For example, above we can see that the sem-lex model apparently tends to embed "HEALTH" and "BRAVE" in ASL Citizen to quite similar vectors, at least by L2 Distance measures

### Precision and Recall
Recall and Precision for each experiment above are [here](https://docs.google.com/spreadsheets/d/1F-JmD7IEOtNU8Tx8KvYeEJKNvlhqDtFNI46GSveO9RY/edit?usp=sharing)

Basically, the model finetuned on the ASL Citizen dataset does best at the ASL Citizen test set. But sem-lex does well too! Nice!


## Notes:

The basic plan, using `pose-format` and `segmentation`, was to do video->pose->segmentation->embeddings, and then search those. 
Similar to https://github.com/sign-language-processing/lexicon-induction

### How to estimate poses?
`videos_to_poses` from `pose-format`
https://github.com/sign-language-processing/pose/blob/master/src/python/pyproject.toml#L61-L64

#### Parallelizing pose estimation
Using [GNU Parallel](https://www.gnu.org/software/parallel/) helped me get this done faster. 

`video_to_pose` will run pose estimation on one video, then exit. But it has to do various overhead things like loading in tensorflow. Doing that every time for every file takes a long time. I ended up doing that anyway at first, because it was still faster that `videos_to_poses`

`videos_to_poses` can reduce the overhead by doing a whole batch. But it still runs one video at a time. If you have 80k videos that takes a long time. This can be done in parallel by splitting a folder of .pose files into a number of subfolders, each with N videos in them. You can do that with a bash script or Python. Then, you do something like:
find <path to folder with many subfolders> -type d -mindepth 1 -maxdepth 1|parallel videos_to_poses --bar --format mediapipe --directory "{}"

### How to segment?
`pose_to_segments` from `segmentation`
* [Demo of segmentation and recognition](https://colab.research.google.com/drive/1CKlXVI3vP0NKZDZZ_I-Qb_wSHt2cw4VT#scrollTo=20_nuF7w3d4N)
* https://github.com/sign-language-processing/segmentation
* https://github.com/sign-language-processing/recognition/blob/main/sign_language_recognition/bin.py cuts but doesn't save

Saving a Pose to a .pose:
```python
    # Write
    print('Saving to disk ...')
    with open(output_path, "wb") as f:
        pose.write(f)
```
TODO: PR to pose-format to document this

### How to embed?
`pose_segment_to_embedding` (not implemented yet)
* SignClip!

### How to search?

Amit Moryossef's advice (which I followed): 
> i’d probably store the embeddings in a database like https://redis.io/solutions/vector-search/ or https://github.com/pgvector/pgvector
then search using the built in database search, similarity, etc. fast stuff

### baseline: search the .poses?
Take the pose vectors and search _those_ based on some distance metric.

## Future work: better models?
* https://github.com/openai/CLIP/issues/83 Wei Chih Chern's training code
