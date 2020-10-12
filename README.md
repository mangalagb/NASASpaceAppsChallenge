The Recommender consists of 3 main files

Scraper - This is responsible for parsing the data fromCSA and extracting relevant
metadata

BuildTDIFModel.py - This is responsible for building the NLP model for
the parsed data

Recommender.py - This is the client facing code that delivers 
the recommendations to the users based on their queries

The data subfolder contains the raw scraped data as well as the model
which is stored for further use.
