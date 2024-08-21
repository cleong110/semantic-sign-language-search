# semantic-sign-language-search
Search a folder of sign language videos semantically.

The basic plan, using `pose-format` and `segmentation`, is to do video->pose->segmentation->embeddings, and then search those. 
Similar to https://github.com/sign-language-processing/lexicon-induction


### How to estimate poses?
`videos_to_poses` from `pose-format`
https://github.com/sign-language-processing/pose/blob/master/src/python/pyproject.toml#L61-L64

### How to segment?
`pose_to_segments` from `segmentation`
* [Demo of segmentation and recognition](https://colab.research.google.com/drive/1CKlXVI3vP0NKZDZZ_I-Qb_wSHt2cw4VT#scrollTo=20_nuF7w3d4N)
* https://github.com/sign-language-processing/segmentation
* https://github.com/sign-language-processing/recognition/blob/main/sign_language_recognition/bin.py cuts but doesn't save

### How to embed?
`pose_segment_to_embedding` (not implemented yet)
* SignClip!

### How to search?
* FAISS maybe?

Amit says: 
> i’d probably store the embeddings in a database like https://redis.io/solutions/vector-search/ or https://github.com/pgvector/pgvector
then search using the built in database search, similarity, etc. fast stuff

### baseline: search the .poses?
Take the pose vectors and search _those_
