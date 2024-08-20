# semantic-sign-language-search
Search a folder of sign language videos semantically.


The basic plan, using `pose-format` and `segmentation`

    videos_to_poses on my data
    pose_to_segments on each .pose
    https://github.com/sign-language-processing/recognition/blob/main/sign_language_recognition/bin.py to cut, similar to https://github.com/sign-language-processing/lexicon-induction
    embed with signCLIP, save off the embeddings
    FAISS maybe?


### How to estimate poses?
`videos_to_poses`
https://github.com/sign-language-processing/pose/blob/master/src/python/pyproject.toml#L61-L64

### How to segment?
`pose_to_segments`
* Demo of segmentation and recognition
* https://github.com/sign-language-processing/segmentation

### How to embed?
`pose_segment_to_embedding` (not implemented yet)
* SignClip!

### How to search?
* FAISS maybe?

Amit says: 
> i’d probably store the embeddings in a database like https://redis.io/solutions/vector-search/ or https://github.com/pgvector/pgvector
then search using the built in database search, similarity, etc. fast stuff
