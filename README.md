## For refactoring 
#### 1. Refactored locally and then tested.
#### 2. Tested file might need to merge by hand.
#### 3. Test the merged file on each branch locally, and merge it to the master.






# graduationproject
#### We thought that game software needs a quick feedback to maintain the software. You can see the most interest of users, through game review category classification. 






# For data collection
#### we crawled reviews from the Google Play Store to collect data
#### we used Selenium 
#### fileName: [local]-crawl.py






# Package
#### We used gensim for word2vec and konlpy for data preprocessing






# Classificate game reviews using word2vec
#### We categorized game reviews as payment, account, configuration, server, system, directing, character, etc.
#### Each category contains about nine sub-words that can represent categories.
#### To find the categories, we internally weighted the matrix and the TDM document
#### fileName: [local]-refactoringCategory.py






# Satisfaction measurement using CNN
#### We classified as satisfaction, normal, and dissatisfaction.
#### fileName: [local]-textcnn.py, train.py, test.py, data.py






# Server setting 
#### We used Amazon Web Services to build the server and Flask web framework.

