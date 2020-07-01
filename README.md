# ML-project
A Project Report
On
H1-B Visa approval using classification
BY
1. Manish Reddy Ami Reddy - 17VE1A05J2
2. Munagala Sainath Reddy - 17VE1A05M0
3. Sindhu Gorli - 317126510020
4. Devi Jayamangala - 17B01A1239
Under the supervision of
Dr.Aruna Malapati
(June 2020)
1
ACKNOWLEDGMENTS
I would like to express my thanks to the people who have helped me most
throughout my project. I am grateful to my trainer Dr.Aruna ma’am and
Goalstreet for nonstop support for the project. A special thank of mine goes to
my team members who helped me out in completing the project, where they
all exchanged their own interesting ideas, thoughts and made this possible to
complete my project with all accurate information. I wish to thank my parents
for their personal support or attention who inspired me to go my own way.
2
ABSTRACT
In our project,our aim is to predict the outcome of H-1B Visa applications
that are applied by many skilled foreign nationals every year. The report addresses
the approach to predict the case status of the filed H1-B Visa petitions using various
data such as employer name, job category, job title, location of job, filing year, and
prevailing wage.We framed the problem as a Classification problem and we applied
the algorithms like Logistic Regression,K-Nearest Neighbours,Decision
Tree,Random Forest and Naïve Bayes to predict the case status of the application that
is applied for H-1B visa.The inputs to our algorithm are the attributes of the
applicant.
H-1B is a type of non-immigrant visa in the United States that allows foreign
countries to work in specified jobs which requires specialized knowledge and a
Bachelor Degree or higher in the specific area.This visa requires the applicant to have
an offer letter from an employer who is from US before they can file an application
to the US Immigration Service(USCIS).USCIS grants 85,000 H-1B visas every
year,even though the number of applicants are more than the count. The selection
process is claimed to be based on a lottery,hence how the attributes of the applicants
affect the final outcome is unclear.
As H-1B visa is one of the highly important one.This approach can be applied
by both individual and the employer in between applying for the visa,so we believe
that this prediction algorithm could be a useful resource both for the future H-1B visa
applicants and the employers who are considering to sponsor them.
3
CONTENTS
Title page……………………………………………………..………….….……..1
Acknowledgements………………………………………………….…….….…...2
Abstract……………………………….……………………………………..…….3
1. Introduction…..………………………….………………...……….….…....5
2. Data Collection..……………………………….……..…………….…...….7
3. Data Representation……………………………..………………….………8
3.1. Data Normalisation……………………………………………….…..11
3.2. Data Visualisation………………………………………………….....14
4. Models………………………………………………………………….…16
4.1. Logistic Regression…………………………………………………...16
4.2. K-Nearest Neighbours…………………………………….…………..18
4.3. Decision Tree……………………………………………….………..21
4.4. Random Forest……………………………….………………………22
4.5. Naive Bayes……………………………………………….…………25
5. Conclusion……………………………………………....………….………27
References…………………………………………….…………………….……..29
4
1. INTRODUCTION
H-1B Visa is the guide of authorization on a travel permit that gives a permit to
the holder to move in, leave or stay in the country for a predetermined timeframe.
There are distinctive kinds of foreigner visas, the required structures, and the means
in the worker visa process contingent upon the nation one needs to move. Moving to
America is a vital and complex decision.
For an outside national to apply for H1B visa, a US business must offer an
occupation and request to for H-1B visa with the US movement office. This is the
most widely recognized visa status connected to and held by universal understudies
once they finish school/advanced education (Masters,Ph.D.) and work in a full-time
position. The help begins the movement methodology by recording an interest to for
the remote inhabitant’s purpose with U.S. Residency and Colonization Facilities
(USCIS). The Office of Foreign Labour Certification (OFLC) creates program
information that is helpful data about the movement programs including the H1-B
visa.
It is intended to carry outside experts with professional educations and
specific aptitudes to fill occupations when qualified Americans can’t be found. Be
that as it may, as of late, worldwide outsourcing organizations have ruled the
program, winning a huge number of visas and pressing out numerous American
organizations, including littler new companies. To take one vital case, in India, the
quantity of first degrees presented in science and designing rose from 176 thousand
of every 1990 to 455 thousand of every 2000. Second, the Act of 1990 set up the
H-1B visa package for impermanent labourers in "claim to fame occupations".
The rules defines "claim to fame occupation" as requiring hypothetical and
common-sense use of a collection of exceptionally particular learning in a field of
human undertaking including, yet not constrained to, design, building, arithmetic,
physical sciences, sociologies, solution and wellbeing, instruction, law, bookkeeping,
business fortes, religious philosophy, and expressions of the human experience.
Furthermore, candidates are required to have achieved a four year
certification or it’s identical as a base. Firms that desire to contract non-natives on
H1B visas have needs to file a Labour Condition Application (LCA) . In LCA’s for
H-1B specialists, the business must bear witness to that the firm will pay the
non-foreigner the more prominent of the genuine remuneration paid to different
5
representatives in a similar activity or the common pay for that occupation, and the
firm will give working conditions to the non- migrant that don’t make the working
states of alternate workers be unfavourably affected.
By then, planned H-1B non-outsiders must exhibit to the US Citizenship and
Immigration Services Bureau (USCIS) in America.In spite of the fact that H1B visa
contributed a considerable measure to the economy of USA by bringing the skilled
non-natives, it additionally influences American work [1]. They lose their
employment, as firms incline toward modest work when contrasted with American’s.
The objective of the H1B program is to connect a work hole in the U.S without
influencing U.S specialists.
At to start with,the structure of H1B is to fill work hole however current
structure encourages businesses to augment the work hole as there aren’t any
qualified U.S specialists and they are procuring modest remote labourers as H1B
program. There are 3 primary targets of the H1B program: Section 1) To connect a
work hole without dislodging U.S specialists forever. Section 2) Review the present
structure of H1B program, concentrating on the way toward acquiring H1B visa and
what it enables its holders to do. It gives two classifications of the run the show:
Qualification of remote specialists.
a) The framework guarantees that outsiders don’t dislodge U.S specialists.
Section 3) Effect of H1B structure on compensation framework. Paying non-natives
not exactly or equivalent to U.S representatives to make a disincentive U.S specialists
So, from a Machine Learning perspective, this challenge posses a possibility
of an underlying pattern that can be identified by training a binary classifier. This
implies, we just need to focus on the target column of our dataset and extract relevant
features that might form a pattern which can be recognised by our classifier.
The main objectives of current work are as follows:
1) Detail investigation of effectively existing machine learning systems and upgraded
Machine Learning approaches for a better forecast.
2) Approve different models in the wake of observing at on premise insights
estimations for using sensible endorsement framework.
6
3) Finally, we prepare a proposed model with the marked data to foresee future
petitions as a right one or mishandle and then validate it by using suitable validation
technique
2. Data Collection
H-1B Visa for the Classifier was acquired from an open Kaggle dataset, a
collection of 12000 articles that span various genres .
By the analysis done by some organisations, U.S receives more than 1,00,000
applications from all nations every year.But many of them are rejected because those
visas are given in lottery method.So our main aim is to build a machine learning
model that predicts the likelihood of H1-B visa application.Hence with this aim in
mind we decided to create a model with H1-B visa petitions 2011-2016 dataset from
the Kaggle.
H-1B Visa database contains approximately 3 million records overall. The
columns in the dataset include case status, employer name, worksite coordinates, job
title, prevailing wage, occupation code, and year filed.
Figure 1 H-1B Visa
The certified data is now clubbed with the non certified news data. The ratio
of the certified to non-certified news was estimated on the basis of empirical
evidence with an attempt to replicated the state of published news in the real world. I
7
used 27,07,835 certified applicants with 85178 non-certified applicants for a total of
around 28,00,000 entries as my complete dataset.
The dataset of H1-B visa is first filtered by removing Not A Number(NaN)
, rows with no entries , Columns which are not useful and the attributes other than
certified which belongs to CASE_STATUS column are to be defined under Denied
status.
After filtering, the certified and non-certified they are clubbed together into a
dataset. This dataset is shuffled and divided into ‘train-dataset’ and ‘test_dataset’ in
the ratio 80%:20%.
3. DATA REPRESENTATION
The dataset that we are using is H-1B visa petitions 2011-2016 which is
derived from the Kaggle dataset. H-1B Visa database contains approximately 3
million records overall.The following are the steps used to solve the problem.
● Read the Input dataset.
● Perform all necessary Data Normalization, Standardization processing
toprepare the transformed format of given input dataset.
● Handle Missing values
● Perform Exploratory Data Analysis/Visualization and bring insights of the
predictor variables.
● Apply Logistic Regression,KNN,Decision Tree,Random Forest and Gaussion
Naïve Bayes algorithms by splitting the data into train and test sets
● Measure and compare the performance of the models using confusion matrix
and metrics like Precision and Recall
● Apply statistical test to explain the goodness of the fit
First five records of the dataset are as follows
8
Figure 2: Head of H1-B visa Dataset
There are total of 11 columns in our dataset each consisting of some specific
records.The values which are stored and meaning of each column are explained
below
● Unnamed :0 Column -It is a not named column,which stores ID of the
row
● CASE_STATUS - It indicates the status of the application
● EMPLOYER_NAME -The name of the employer as registered in H1-B
visa application
● SOC_NAME - The occupation code for the employment
● JOB_TITLE - The job title for the employment
● FULL_TIME_POSITION - Indicates whether the application is for full time or
for the part time employment
● PREVAILING_WAGE - The most frequent wage for a corresponding role
as filled in the visa application
● YEAR -The application year
● WORKSITE -The address of the employer worksite
● LON -Longitude of employer worksite
● LAT -Latitude of employer worksite
9
Description of each column of the dataset is given below:
Figure 3 : Description of H1-B visa Dataset
In this dataset our target column is CASE_STATUS as it consists of the
information whether a visa is certified or not.And in the CASE_STATUS there are 6
unique values that represents whether an application is approved or not.Those unique
values are as follows.
10
Figure 4: Values of CASE_STATUS column from H1-B visa Dataset
1.Data Normalisation:
Then we performed the data normalisation in which the main aim is to
remove the unwanted or unusable entries or unusable columns.At first we merged all
the records other than Certified are merged under the Denied status and then we
removed LAT and LON columns.Then we created a new column called as
NEW_EMPLOYER and then we dropped EMPLOYER_NAME,Worksite and
Unnamed:0 so as to normalise the data.The above process is shown below.
Figure 5 : Considering all values other than Certified as Denied
11
 Figure 6: Dropping LAT and LON columns
Figure 7 : Replacing Values with other one
12
Figure 8 : Splitting city and state and capturing state to another variable
Figure 9: Head of dataset after Normalisation
13
2.Data Visualization: Then we performed data visualisation between the columns
of the dataset and we plotted a histogram as follows
Figure 10: Histogram drawn for columns
14
Then we plotted the countplot between CASE_STATUS and count to know how
many applications are certified and how many are rejected.The resultant graph is as
follows:
Figure 11: Count plot drawn to know how many applications accepted or not
We can see that there are more number of certified class like more than 25,00,000
and there are very few denied class records,which tells that certified visa applications
are much more than rejected visa applications.
15
4.Models
Several different Classifying Models were implemented to accurately
predict if the H-1B visa for an applicant is cerfied or not certified.The training dataset
and test dataset from above data preprocessing is used to train and check the
reliability of our model. The Accuracy and F1 Scores on the test set are reported for
each model. Here, the F1 Score is a more reliable testing parameter as the ratio of the
Certified to Non-Certified is not 0.5, rather it was estimated on the basis of empirical
evidence to replicate the state of published news in the real world.
1. Logistic Regression
A Logistic Regression algorithm is used to create a binary classifier that is
optimised on our training dataset. The Logistic Regression function from sklearn
library is used to create and train our classifier. We did not used any parameters of
the function.
The model was trained after dimensionality reduction. Then the model was
tested for accuracy on the test dataset and a Confusion Matrix was plotted along with
test Accuracy, Precision, Recall and F1Score was reported.
Results :
The training and then prediction for the test set was performed using the
same total dataset. For every iteration the dataset was split randomly into Training
and Test set then the dimensionality reduction was carried out on the training and the
test set. The model performance in terms of the Accuracy, Precision. Recall and F1
Score was recorded and the final results are as follows.
16
Figure 12 :Precision,Recall,F1 score,Confusion matrix,Accuracy for Logistic Regression without Undersampling
F1 score,Recall and Confusion Matrix for Certified status is very low when
compared to Non Certified.So we performed UnderSampling using NearMiss
algorithm and the results are as follows
17
 Figure 13 :Precision,Recall,F1 score,Confusion matrix,Accuracy for Logistic Regression with Undersampling
After performing the UnderSampling the precision,recall,F1 Score for
Non-Certified increased exponentially but in case of Certified there is decrease in
above parameters.And the total accuracy of this model is dropped from 97% to
68%.So this algorithm is not efficient to work with.So we performed K-Nearest
Neighbours algorithm.
2. K-Nearest Neighbors Algorithm
K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm
which can be used for both classification as well as regression predictive
problems.KNN algorithm assumes the similarity between the new case/data and
available case and put the new case into the category that is most similar to the
available categories.KNN algorithm stores all the available data and classifies a new
18
data point based on the similarity.This means when new data appears then it can be
easily classified into a well suite category by using KNN algorithm.KNN is
non-parametric algorithm,which means it does not makes any assumptions on
underlying data.It is also called as a Lazy Learner Algorithm because it does not
learn from the training set immediately instead it stores that dataset and at the time of
classification,it performs an action on the dataset.KNN algorithm at the time of the
training,it just stores the dataset and when it gets the new data,then it classifies that
data into a category that is much similar to the new data.The parameters of the
function used :
● n_neighbors: It specifies the number of neighbors to use by default for
k-neighbors queries . It takes integer values as parameter and the default
value=5
To find the exact k-value we applied the elbow method and plotted the graph for
different k values vs Error rate
 Figure 14 : Plot of error rate vs K-Value
19
By looking at the plot we can observe that at k=3 the error rate is very low,so we
train the knn algorithm by n_neighbors=3.
Results :
The training and then prediction for the test set was performed using the same
total dataset. For every iteration the dataset was split randomly into Training and Test
set then the dimensionality reduction was carried out on the training and the test set.
The model performance in terms of the Accuracy, Precision, Recall and F1 Score are
as follows.
Figure 15 :Precision,Recall,F1 score,Confusion matrix,Accuracy for KNN
20
3. Decision Tree
Decision Trees (DTs) are a non-parametric supervised learning method
used for classification and regression. The goal is to create a model that predicts the
value of a target variable by learning simple decision rules inferred from the data
features. A decision tree is a flowchart-like structure in which each internal node
represents a “test” on an attribute (e.g. whether a coin flip comes up heads or tails),
each branch represents the outcome of the test, and each leaf node represents a class
label (decision taken after computing all attributes). The paths from root to leaf
represent classification rules.
Tree based learning algorithms are considered to be one of the best and mostly
used supervised learning methods. Tree based methods empower predictive models
with high accuracy, stability and ease of interpretation. Unlike linear models, they
map non-linear relationships quite well. They are adaptable at solving any kind of
problem at hand (classification or regression). The parameters of the function used :
● Criterion: The function to measure the quality of a split. Supported criteria are
“gini” for the Gini impurity and “entropy” for the information gain.
It takes these parameters:{“gini”, “entropy”}, default=”gini”
Result :
The training and then prediction for the test set was performed using the same total
dataset. For every iteration the dataset was split randomly into Training and Test set
then the dimensionality reduction was carried out on the training and the test set. The
model performance in terms of the Accuracy, Precision. Recall and F1 Score was
recorded and the final results are as follows.
21
Figure 16 :Precision,Recall,F1 score,Confusion matrix,Accuracy for Decision Tree Classifier
4. Random Forest
The random forest algorithm creats a forest with a number of Decision Trees.
It is a type of Ensemble machine learning algorithm, which use a
divide-and-conquer approach. The main principle behind ensemble algorithms is
boosting, that is a group of weak learners (single estimator or a decision tree) can
work together to form a strong learner (group of estimators or a forest) to classify the
data. The random decision forests can correct for the decision trees’ habit of
overfitting to the training dataset. Hence, random forest algorithm comprises of
bagging (Bootstrap aggregating), which is the approach to reduce overfitting by
combining the classifications of randomly generated training sets, together with the
random selection of features to construct a collection of decision forests.
The Random Forest Classifier is created using the RandomForestClassifier function
of sklearn library. The parameters of the function used :
● n_estimators : The number of decision trees in the forest, We selected 11.
22
The model was trained using the tfidf vector after dimensionality reduction. The
model can also show the feature importance of all the tokens based on the gini
importance criterion.
The feature importance graph is similar to the feature variance plot after the
dimensionality reduction.
Then the model was tested for accuracy on the test dataset and a Confusion Matrix
was plotted along with test Accuracy, Precision, Recall and F1Score was reported.
Results :
The training and then prediction for the test set was performed using the same
total dataset. For every iteration the dataset was split randomly into Training and Test
set then the dimensionality reduction was carried out on the training and the test set.
The model performance in terms of the Accuracy, Precision, Recall and F1 Score are
as follows.
Figure 17 :Precision,Recall,F1 score,Confusion matrix,Accuracy for Random Forest
23
Then we applied Grid Search to the above model and we got the best parameters as
follows:
Figure 18 :Precision,Recall,F1 score,Confusion matrix,Accuracy for Random Forest after applying GridSearch and Elbow method
Then Accuracy, Precision, Recall and F1 Score after applying above methods
are as follows:
24
Figure 19 : Final Precision,Recall,F1 score,Confusion matrix,Accuracy for Random Forest
5. Naive Bayes
A Naive Bayes classifier is a probabilistic machine learning model that’s used
for classification task. The crux of the classifier is based on the Bayes theorem.
P(A/B) = (P(B/A)P(A)) / P(B)
Using Bayes theorem, we can find the probability of A happening, given that B
has occurred. Here, B is the evidence and A is the hypothesis. The assumption made
here is that the predictors/features are independent. That is presence of one particular
feature does not affect the other. Hence it is called naive.
This model was trained using the tfidf vector after dimensionality reduction.
Then the model was tested for accuracy on the test dataset and a Confusion Matrix
was plotted along with test Accuracy, Precision, Recall and F1Score was repored.
Result :
The training and then prediction for the test set was performed 10 times using the
same total dataset. For every iteration the dataset was split randomly into Training
25
and Test set then and dimensionality reduction was carried out on the training and the
test set. The model performance in terms of the Accuracy, Precision, Recall and F1
Score are as follows.
Figure 20 :Precision,Recall,F1 score,Confusion matrix,Accuracy for Naïve Bayes
The accuracy scores of all the classification models we performed are as follows:
26
5.Conclusion
The aim of this project was to test different data representation methods and train
various models using this data. We collected our real data from sources like H-1B
Visa Petitions 2011-2016 dataset was aquired from Kaggle.The resulting transformed
data was trained on the models mentioned in Section 4, and the following results
were extrapolated from this experiment.
Figure 21 :Precision,Recall,F1 score of all algorithms applied
The ratio of Certified to Non-Certified in the dataset were biased to resemble the
real world scenario. Hence, in situation like these the model accuracy is really not
the best metric to base our results. We generally test the performace of a model
trained a biased dataset using F1-Score, which is nothing but the harmonic mean of
precision and recall. The above scores are given for prediction of the fake news
article.
We observe that the Random Forest performed the best in accurately
predicting the visa certified status using a TF-IDF data representation giving a
F1-Score of 98.0%. A close second best performer was the Decision Tree model with
an F1-Score of 98.0% with slight difference in Accuracy.
To know that which model is performing better,we performed the
statistical test for 4 iterations in which we had compared between Random Forest and
Decision Tree by using McNemar test and the results are as follows:
27
Figure 22 :Statistical test between RandomForest and Decision Tree using McNemar Test
As compared to the p-value’s in different iterations,the values are mostly more than
the significant threshold value which is 0.05,So we cannot reject the null hypothesis
and assume that there is no significant difference between the two predictive models.
28
So now based on performing the Random forest and Decision tree for four iterations
we got the accuracy results as follows:
Figure 23 :Mean accuracy of RandomForest and Decision Tree.
Here we observe that the Random Forest performed the best in accurately predicting
the visa status with a mean accuracy of 0.97809 compared to Decision tree’s mean
accuracy of 0.97607.
References
1. Kaggle, H1-B visa petitions, https://www.kaggle.com/nsharan/h-1b-visa, URL
obtained on August 28, 2017.
2. “https://scikit-learn.org/stable/” A website for Machine Learning Information
3. “https://imbalanced-learn.readthedocs.io/en/stable/api.html” A website
refered to perform Undersampling in Logistic Regression
4. “http://rasbt.github.io/mlxtend/” A website refered to perform Statistical Test
29
