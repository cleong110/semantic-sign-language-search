Setup: 
https://github.com/cleong110/semantic-sign-language-search/issues/6
```
apt install libpq-dev
pip install psycopg2
apt install postgresql 
createdb <db_name>
psql "$db_name" -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

I want to be able to: 

1. Given a folder of embedded .mp4 and .npy, add to the db 
2. Given a path (or paths), search for matches


GENERATING A DIAGRAM
I used eralchemy
```
# pip install eralchemy[graphviz]
# sudo apt install graphviz
eralchemy -i "postgresql://vlab@/mydb" -o semantic_sign_language_search_db_schema.er
eralchemy -i semantic_sign_language_search_db_schema.er -o semantic_sign_language_search_db_schema.pdf
```

DONE:
To setup a DB and search all against all

### Setup db and add videos

```
database_name="foo_db"
createdb "$database_name"
psql mydb
CREATE EXTENSION vector;

# setup the structure
python embedding_db.py --recreate 

# add videos with embeddings, using the folder name as the "dataset name"
python add_videos_with_embeddings_to_db.py  /home/vlab/data/ASL_Citizen/ASL_Citizen/ASL_Citizen_curated_sample_with_embeddings_from_all_models/

# add videos with embeddings, specify "dataset_name" so it doesn't get set to "videos"
python add_videos_with_embeddings_to_db.py  /home/vlab/data/ASL_Citizen/ASL_Citizen/asl_citizen_400_words_10_examples_each/videos --dataset_name "asl_citizen_400_words_10_examples_each"


```

### searching the db and calculate metrics
```
# list models and datasets in database
python search_db.py --list_embedding_models --list_datasets


# search, list models, list datasets, search all against all, restrict model
embedding_model="asl-signs" && python search_db.py --list_embedding_models --list_datasets --search_all_against_all -n 10 --pose_embedding_model_for_search "$embedding_model" > "search_results_20_words_5_examples_$embedding_model.txt"

# ASL_Citizen_full 
embedding_model="baseline_temporal" && python search_db.py --list_embedding_models --list_datasets --search_all_against_all -n 10 --search_model "$embedding_model" --search_dataset "ASL_Citizen_full" > "search_results_asl_citizen_full_set_$embedding_model.txt"

########################################
# 400x10 set
# asl_citizen_400_words_10_examples_each, models asl-signs sem-lex baseline_temporal asl-citizen"
embedding_model="asl-signs" && python search_db.py --list_embedding_models --list_datasets --search_all_against_all -n 10 --search_model "$embedding_model" --search_dataset "asl_citizen_400_words_10_examples_each" > "search_results_asl_citizen_400_words_10_examples_each_$embedding_model.txt"

embedding_model="sem-lex" && python search_db.py --list_embedding_models --list_datasets --search_all_against_all -n 10 --search_model "$embedding_model" --search_dataset "asl_citizen_400_words_10_examples_each" > "search_results_asl_citizen_400_words_10_examples_each_$embedding_model.txt"

embedding_model="baseline_temporal" && python search_db.py --list_embedding_models --list_datasets --search_all_against_all -n 10 --search_model "$embedding_model" --search_dataset "asl_citizen_400_words_10_examples_each" > "search_results_asl_citizen_400_words_10_examples_each_$embedding_model.txt"

embedding_model="asl-citizen" && python search_db.py --list_embedding_models --list_datasets --search_all_against_all -n 10 --search_model "$embedding_model" --search_dataset "asl_citizen_400_words_10_examples_each" > "search_results_asl_citizen_400_words_10_examples_each_$embedding_model.txt"


find . -type f -name "search_results_asl_citizen_400_words_10_examples_each_*"| parallel "echo;echo '===>{/}<======';head -n 14 {}&& tail -n 2 {}" > results/asl_citizen_400_words_10_examples_each.txt


##############################################


# test, specifying there are known to be 4 correct answers per class, retrieving 10 each time. 
python search_db.py --search_all_against_all -n 10 -K 4

# test, but don't specify correct answer count and let it figure it out from the glosses
python search_db.py --search_all_against_all -n 10

python search_db.py --search_all_against_all -n 10

# output to both
python search_db.py -n 10 -K 4 2>&1 | tee out.txt

# output to file only 
python search_db.py -n 10 -K 4 2>&1 > search_results_400_words_10_examples.txt

```


### peewee resources

#### Query examples
https://docs.red-dove.com/peewee/peewee/query_examples.html 

### old
```

# output to both
python create_sqlite_peewee.py ./ASL_Citizen_curated_sample_embedded_with_signCLIP/videos/ "ASL Citizen Curated Sample Embedded with Signclip Temporal" --pose_embedding_model "signclip_asl-citizen" --recreate 2>&1 | tee out.txt

# output to file only 
python create_sqlite_peewee.py asl_citizen_400_words_10_examples_each/videos/ asl_citizen_400_words_10_examples --search_all_against_all --recreate 2>&1 > search_results_400_words_10_examples.txt
```




See also https://github.com/cleong110/semantic-sign-language-search/issues/6
To setup the database for searching embedding I did: 


* setup a postgresql server https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-20-04
* `apt libpq-dev apt install`
* `apt install postgresql`
* `pip install psycopg`
* `pip install peewee`
* `pip install pgvector`, and perhaps `sudo apt install postgresql-server-dev-14`
* create a db: `createdb mydb`, you can use `psql` 
* enable the pgvector extension, something like 
```
psql -d mydb
psql (14.13 (Ubuntu 14.13-0ubuntu0.22.04.1))
Type "help" for help.

mydb=# CREATE EXTENSION IF NOT EXISTS vector
```


Random guess chances are: 

formula for calculating: 
My probability skills are a little rusty, but I think this is an "unordered sampling without replacement" problem. The formula for the expected number of matches per search would be: E[matches per search]=1∗P(1 match per search)+2∗P(2 matches per search)+3∗P(3 matches per search)+4∗P(4 matches per search)E[matches per search]=1∗P(1 match per search)+2∗P(2 matches per search)+3∗P(3 matches per search)+4∗P(4 matches per search)

Where P(n matches per search)=(4 choose n)(95 choose 10-n)/(99 choose 10)P(n matches per search)=(4 choose n)(95 choose 10-n)/(99 choose 10). Basically the number of cases where there are n matches divided by the total number of possibilities (99 choose 10). 

simulated:
```
(peewee) vlab@vlab-desktop:~/projects/semantic-sign-language-search/setup_db$ python random_guess_retrieval_simulation.py
Simulating 400 classes with 10 examples each: Running 10000 trials
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [01:24<00:00, 117.80it/s]After running 10000 trials, the mean of the mean match counts was: 0.02244
(peewee) vlab@vlab-desktop:~/projects/semantic-sign-language-search/setup_db$ vi random_guess_retrieval_simulation.py
(peewee) vlab@vlab-desktop:~/projects/semantic-sign-language-search/setup_db$ python random_guess_retrieval_simulation.py
Simulating 400 classes with 10 examples each: Running 10000 trials
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [01:22<00:00, 120.54it/s]After running 10000 trials, the mean of the mean match counts was: 0.022289999999999997
```

