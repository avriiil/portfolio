![cover_photo](./readme_files/tweets_cover.jpg)
# What the Tweet?! -- Identifying Arabic-Language Political Misinformation on Twitter
**Using unsupervised NLP topic modelling and clustering to build a machine learning classifier that can identify misinformation in short-text documents.**

## Introduction
The internet — and social media, especially — is rife with political misinformation. Besides being a potential nuisance when scrolling through your Twitter feed, this misinformation can also have dramatic real-world repercussions, like the January 6 storming of the U.S. Capitol. For this reason, detecting political misinformation and hate speech is a major challenge for organisations in the political, humanitarian and intelligence sectors. One of the main challenges in fighting misinformation is the lack of labelled datasets, especially in low-resource languages.

After speaking to several people involved in misinformation and hate speech detection across a range of organisations, I decided to use my final Capstone Project of the Springboard Data Science Career Track to address some of these issues. With this project, I hope to contribute to the ongoing research in this field by looking specifically at the tools and processes necessary to detect political misinformation in Arabic-language tweets.

## The Data
For this project, I make use of a dataset from Twitter’s [Transparency Center]9https://transparency.twitter.com/en/reports/information-operations.html) consisting of 5,350 Twitter accounts (and all of their 36+ million tweets, 95 percent of which are in Arabic) that have been identified by Twitter as being part of state-linked Information Operations.

In Twitter’s own words:

“A network of accounts associated with Saudi Arabia and operating out of multiple countries including KSA, Egypt and UAE, were amplifying content praising Saudi leadership, and critical of Qatar and Turkish activity in Yemen. A total of 5,350 accounts were removed.”

The data was downloaded on February 26, 2021.

## The Approach
An important thing to note here is that the accounts were identified as ‘compromised’ and not every single one of its Tweets, per se. This meant that it wasn’t a given that all of the 36M tweets actually contained political misinformation.

The goal of the project, then, was to figure out how to boil the 36+ million tweets down to the actual political misinformation content.

I do this in 3 steps:
1. *Wrangling*: Clean the data to get only the unique tweets.
2. *Topic Modelling*: Perform NLP-driven categorisation of the unique tweets to sift out all the end-of-the-workday cat memes and other junk and get only the political content.
3. *Clustering*: Apply unsupervised learning methods to further distinguish between political content and political misinformation.

Which then prepares us for the final step:
4. *Classification*: Using the output to construct an ML classifier.

![workflow_1](./readme_files/approach_flow.png)

![workflow_2](./readme_files/workflow.png)




