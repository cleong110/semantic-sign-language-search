To setup a DB and search all against all

```
python create_sqlite_peewee.py ./ASL_Citizen_curated_sample_embedded_with_signCLIP/videos/ "ASL Citizen Curated Sample Embedded with Signclip Temporal" --pose_embedding_model "signclip_asl-citizen" --recreate 2>&1 | tee out.txt
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
