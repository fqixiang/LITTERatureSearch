# Meeting 20210108
[The presentation](https://docs.google.com/presentation/d/1EIYqbjCgD7ClhMcrJbf8yekJ5Ap-mRKtUY0Fkku5QLU/edit?usp=sharing)

## Feedback from the challenge owners
### Search terms
* important additional terms: nanoplastic, macro plastic
* less important: mesoplastic, contamination (too many items perhaps)
* leave out toxicants like PBDE

### Database
* journals like marine pollution bulletin are important to be included

### Pipeline
* add step 5: remove duplicate data

### Labelled data
* should Sophie Schmiz on the data collection process and the final available data
* [Mikael's paper](https://pubs.acs.org/doi/full/10.1021/acs.est.0c01984) points to 17 relevant papers

### Other
* the expected time window of sampling is after 2008
* the expected number of relevant papers is in the hundreds (between 10 and 1000)
* the problem with LITTERBASE is that it is handcurated, but at some point we can contact them about using their findings as gold standard
* the ultimate goal is to automate as much as possible

## Next steps
- [ ]  Make an appoint with Sophie to learn about her data
- [ ]  Identity those relevant papers and curate a data set (titles, abstracts and DOIs) - Qixiang 
- [ ]  Get a large collection of papers from databases (with export information like titles, abstracts and DOIs) - Sebastian
- [ ]  Write a script to crawl meta data about the articles using the DOIs - Mehran & Qixiang
- [ ]  Write a script or find a way to automate batch download of the full texts - Sebastian & Mehran
- [ ]  Write a model to identify relevant papers - later
- [ ]  See if it's possible to make use of ASReview - later


# Kick-off Meeting 20201222

## The ultimate goal (not of relevance to the current challenge)
* A 3d map of plastics in the ocean
* Eventually a virtual simulation of how plastics move
* Need as many observations of plastics as possible (to train the model)

## Pre-existing work
* 2 master students going through literature manually
* The goal of the challenge is scraping of the literature
* There is a small database of the Mediterranean ocean now - curated data set ready by mid January


## Main challenges
* How to automatically identify peer-reviewed papers that contain data on observations of plastic in the ocean and on beaches>
* How to automatically parse that data into a database?
* How to geo-tag the plastic observations to location and time of sampling?

## Tips

* Start with challenge 1: identify relevant papers and if possible, automate extraction of the data sets
* Start with more recent papers: look for data availability statement
* We can get some expected meta data from Darshika.
* Darshika is first point of contact for any questions

## To Do:

- [x]  Make a meeting on Jan. 8
- [x]  Ask for a couple of papers
- [x]  Start with Challenge 1
- [ ]  Expect meta data afrom Darshika
