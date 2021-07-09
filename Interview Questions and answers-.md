# Interview Questions and answers:-


## What is machine learning?

- Machine learning is a branch of AI focused on building applications that learn from data and improve their accuracy over time without being programmed to do so. 

## What is the relation between Al, ML & DL?

![img](https://www.aimlmarketplace.com/images/Startup-images-1/difference-between-ai-ml-dl.png)



## What are the different types of machine learning ? Explain each of them.

- These are three types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

  - **Supervised Learning** is one of the most basic types of machine learning. In this type, the machine learning algorithm is trained on labelled data.

  - **Unsupervised machine learning** holds the advantage of being able to work with unlabelled data. This means that human labour is not required to make the dataset machine-readable, allowing much larger datasets to be worked on by the program.

  - **Reinforcement Learning** directly takes inspiration from how human beings learn from data in their lives. It features an algorithm that improves upon itself and learns from new situations using a trial-and-error method. Favourable outputs are encouraged or ‘reinforced’, and non-favourable outputs are discouraged or ‘punished’.

  - ##### The followup question can be from any of the 3 types.

    


## What is machine learning algorithms?

- Machine learning algorithms are programs (maths and logic) that adjust themselves to perform better as they are exposed to more data. The “learning” part of machine learning means that those programs change how they process data over time, much as humans change how they process data by learning.
- In machine learning, algorithms are 'trained' to find patterns and features in massive amounts of data in order to make decisions and predictions based on new data. The better the algorithm, the more accurate the decisions and predictions will become as it processes more data.

##  What are the differences between supervised and unsupervised learning?

|                     Supervised Learning                      |                    Unsupervised Learning                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|            Uses known and labelled data as input.            |                Uses unlabelled data as input.                |
|        Supervised learning has a feedback mechanism.         |       Unsupervised learning has no feedback mechanism.       |
| The most commonly used supervised learning algorithms are decision trees, logistic regression, and support vector machine. | The most commonly used unsupervised learning algorithms are k-means clustering, hierarchical clustering, and apriori algorithm. |



### Tomorrow we will see some algorithm specific questions on supervised learning.

### Please post your questions in the comment section, I will answer it, tomorrow.


### There are lots of basic and advance questions are there which will be coming up, stay tuned.



# Notes of today's Project.

## Breast Cancer type prediction.

### Problem Statement

- We have the dataset of the patients with breast cancer, With it's analysis we can predict and diagnose patients, and generate the report which will show which type of breast cancer they are suffering from and also recommend the immediate actions to be taken on the basis of their cancer type.


### Data set description

- Download dataset from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

#### Attribute Information:

- ID number
- Diagnosis (M = malignant, B = benign)

- Ten real-valued features are computed for each cell nucleus:
  - Radius (mean of distances from centre to points on the perimeter)
  - Texture (standard deviation of Gray-scale values)
  - Perimeter
  - Area
  - Smoothness (local variation in radius lengths)
  - Compactness (perimeter^2 / area - 1.0)
  - Concavity (severity of concave portions of the contour)
  - Concave points (number of concave portions of the contour)
  - Symmetry
  - Fractal dimension ("coastline approximation" - 1)

- The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image,resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

- All feature values are recorded with four significant digits.

- Missing attribute values: none

- Class distribution: 357 benign, 212 malignant

## Which one is your favourite ML algorithm?

### or 

## Choose any ML algorithm which you know best?

- Answer of this question is completely of your choice but which ever you choose you should know each and every part of it.
- The follow up question will be on that particular algorithm.
- Some ML algorithms is discussed below.

## What are the different type of Regression Algorithms?

- Linear Regression
- Logistic Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Bayesian Linear Regression

## What is linear regression?

- In simple words in linear regression we try to fit a linear straight line between feature(independent) and target(dependent) variable as a relationship. We use it when the target variables are continuous.
- Our aim is to keep the error(Root Mean Squared Error) minimum while fitting the line.

## What are the assumptions of linear regression?

- Linear relationship : There should be a linear and additive relationship between dependent (response) variable and independent (predictor) variable(s). A linear relationship suggests that a change in response Y due to one unit change in X¹ is constant, regardless of the value of X¹. An additive relationship suggests that the effect of X¹ on Y is independent of other variables.

- No or little multi-col-linearity : The independent variables should not be correlated. Absence of this phenomenon is known as multi-col-linearity.

- No auto-correlation : There should be no correlation between the residual (error) terms. Absence of this phenomenon is known as Auto-correlation.

- Homoscedasticity :The error terms must have constant variance. This phenomenon is known as homoscedasticity. The presence of non-constant variance is referred to homoscedasticity.

- Multivariate normality : The error terms must be normally distributed.

#### How to deal with all this if it is not there will be discussed on some other day.

## What is Random Forest ?

- Random Forest or random decision forests are a learning method for classification, regression by making multiple decision tree using random samples from training data.

## What does random mean in Random forest?

- Random forest adds additional randomness to the model, while growing the trees. Instead of searching for the most important feature while splitting a node, it searches for the best feature among a random subset of features.

### What are the steps to follow while implementing Random Forest.

- Take the original data-set and take random sub-datasets.
- Implement decision tree on the random sub-datasets.
- As we get different predictions from different decision tree we need to take the result which is most relevant.
- For taking the most appropriate result we need to use different methods for different type of problems:
  - For classification we take prediction which has been given by maximum no of decision trees.
  - For regression as we have got a continuous value we will take the average of the predictions given by the decision trees.

## What is KNN(K-Nearest Neighbour)?

- KNN is a supervised ML Algorithm that performs both classification and regression tasks using the numbers(K) of neighbours(Nearest).

## What is K in KNN?

'K' in KNN is a parameter that refers to the number of nearest neighbours to include in the majority of the voting process.

### Steps to implement KNN:

- Getting data.
- Defining K Neighbours.
- Calculating the neighbour's distance.
- Assigning new instance to majority of neighbours.

#### How to calculate distance.

- Euclidean Distance.

  - The Euclidean distance between two points in Euclidean space is the length of a line segment between the two points. It can be calculated from the Cartesian coordinates of the points using the Pythagorean theorem, therefore occasionally being called the Pythagorean distance.

- Manhattan distance.

  - The distance between two points measured along axes at right angles. In a plane with p1 at (x1, y1) and p2 at (x2, y2), it is |x1 - x2| + |y1 - y2|. 

   

# Notes of today's Project.

### Problem Statement

- To Track COVID-19 vaccination in the World, answer instantly to your questions:
- Which country is using what vaccine?
- In which country the vaccination programme is more advanced?
- Where are vaccinated more people per day? But in terms of percent from entire population ?

### Data set description

- Data is collected daily from Our World in Data GitHub repository for covid-19, merged and uploaded.

#### Attribute Information:

- Country- this is the country for which the vaccination information is provided.

- Country ISO Code - ISO code for the country.

- Date - date for the data entry; for some of the dates we have only the daily vaccinations, for others, only the (cumulative) total.

- Total number of vaccinations - this is the absolute number of total immunisations in the country.

- Total number of people vaccinated - a person, depending on the immunisation scheme, will receive one or more (typically 2) vaccines; at a certain moment, the number of vaccination might be larger than the number of people.

- Total number of people fully vaccinated - this is the number of people that received the entire set of immunisation according to the immunisation scheme (typically 2); at a certain moment in time, there might be a certain number of people that received one vaccine and another number (smaller) of people that received all vaccines in the scheme.

- Daily vaccinations (raw) - for a certain data entry, the number of vaccination for that date/country.

- Daily vaccinations - for a certain data entry, the number of vaccination for that date/country.

- Total vaccinations per hundred - ratio (in percent) between vaccination number and total population up to the date in the country.

- Total number of people vaccinated per hundred - ratio (in percent) between population immunised and total population up to the date in the country.

- Total number of people fully vaccinated per hundred - ratio (in percent) between population fully immunised and total population up to the date in the country.

- Number of vaccinations per day - number of daily vaccination for that day and country.

- Daily vaccinations per million - ratio (in ppm) between vaccination number and total population for the current date in the country.

- Vaccines used in the country - total number of vaccines used in the country (up to date).

- Source name - source of the information (national authority, international organisation, local organisation etc.).

- Source website - website of the source of information.


# Summery 

## Covaxin is used globally.

## In India Total Vaccination and People Vaccination rate is very high.

## India Secured 4th rank in Globally Total Vaccination Country list .

## India Secured 3rd rank in Globally Daily Vaccination Country list.







## What is Decision Tree algorithm?

- A decision tree is a popular supervised machine learning algorithm. It is mainly used for Regression and Classification. 
- It allows breaks down a dataset into smaller subsets.
- The decision tree can able to handle both categorical and numerical data.
- A decision tree is a tree in which every node specifies a test of some attribute of the data and each branch descending from that node corresponds to one of the possible values for this attribute.

## To which kind of problems are decision trees most suitable?

- Decision trees are most suitable for tabular data.
- The outputs are discrete.
- Explanations for decisions are required.
- The training data may contain errors.
- The training data may contain missing attribute values.

## On what basis is an attribute selected in the decision tree for choosing it as a node?

- Attribute selection is done using Information Gain in decision trees. The attribute with maximum information gain is selected.

## What is Information Gain? What are its disadvantages?

- Information gain is the reduction in entropy due to the selection of an attribute. 
- Information gain ratio biases the decision tree *against* considering attributes with a large number of distinct values which might lead to overfitting. 
- In order to solve this problem, information gain ratio is used.

## What is the inductive bias of decision trees?

- Shorter trees are preferred over longer trees. Trees that place high information gain attributes close to the root are preferred over those that do not.

## How does a decision tree handle continuous attributes?

- By converting continuous attributes to a threshold-based Boolean attribute. The threshold is decided by maximising the information gain.

## How does a decision tree handle missing attribute values?

- One way to assign the most common value of that attribute to the missing attribute value. The other way is to assign a probability to each of the possible values of the attribute based on other samples.

## What are the different types of nodes in Decision Trees.

- The Decision Tree consists of the following different types of nodes:
  - **1. Root node:** It is the top-most node of the Tree from where the Tree starts.
  - **2. Decision nodes:** One or more Decision nodes that result in the splitting of data into multiple data segments and our main goal is to have the children nodes with maximum homogeneity or purity.
  - **3. Leaf nodes:** These nodes represent the data section having the highest homogeneity.

##  What Pruning in Decision Trees? Why be we do it?

- After we create a Decision Tree we observe that most of the time the leaf nodes have very high homogeneity i.e., properly classified data. However, this also leads to overfitting. Moreover, if enough partitioning is not carried out then it would lead to underfitting.

- Hence the major challenge that arises is to find the optimal trees which result in the appropriate classification having acceptable accuracy. So to cater to those problems we first make the decision tree and then use the error rates to appropriately prune the trees.

## **What do you understand by Pruning in a Decision Tree?**

When we remove sub-nodes of a Decision node, this process is called pruning or the opposite process of splitting. The two techniques which are widely used for pruning are- Post and Pre Pruning.

**Post Pruning:**

- This type of pruning is used after the construction of the Decision Tree.
- This technique is used when the Decision Tree will have a very large depth and will show the overfitting of the model.
- It is also known as backward pruning.
- This technique is used when we have an infinitely grown Decision Tree.

**Pre Pruning:**

- This technique is used before the construction of the Decision Tree.
- Pre-Pruning can be done using Hyperparameter tuning.
- Overcome the overfitting issue.

## **List down some popular algorithms used for deriving Decision Trees along with their attribute selection measures.**

- Some of the popular algorithms used for constructing decision trees are:
  - **1.** **ID3 (Iterative Dichotomiser):** Uses Information Gain as attribute selection measure.
  - **2.** **C4.5 (Successor of ID3):**  Uses Gain Ratio as attribute selection measure.
  - **3. CART (Classification and Regression Trees)** – Uses Gini Index as attribute selection measure.

## What are the advantages of the Decision Trees?

**1. Clear Visualization:** This algorithm is simple to understand, interpret and visualize as the idea is mostly used in our daily lives. The output of a Decision Tree can be easily interpreted by humans.

**2. Simple and easy to understand:** Decision Tree works in the same manner as simple if-else statements which are very easy to understand.

**3.** This can be used for both classification and regression problems.

**4.** Decision Trees can handle both continuous and categorical variables.

**5. No feature scaling required:** There is no requirement of feature scaling techniques such as standardisation and normalization in the case of Decision Tree as it uses a rule-based approach instead of calculation of distances.

**6. Handles nonlinear parameters efficiently:** Unlike curve-based algorithms, the performance of decision trees can’t be affected by the Non-linear parameters. So, if there is high non-linearity present between the independent variables, Decision Trees may outperform as compared to other curve-based algorithms.

**7.** Decision Tree can automatically handle missing values.

**8.** Decision Tree handles the outliers automatically, hence they are usually robust to outliers.

**9. Less Training Period:** The training period of decision trees is less as compared to ensemble techniques like Random Forest because it generates only one Tree unlike the forest of trees in the Random Forest.

## What are the disadvantages of the Decision Trees?

**1. Overfitting:** This is the major problem associated with the Decision Trees. It generally leads to overfitting of the data which ultimately leads to wrong predictions for testing data points. it keeps generating new nodes in order to fit the data including even noisy data and ultimately the Tree becomes too complex to interpret. In this way, it loses its generalization capabilities. Therefore, it performs well on the training dataset but starts making a lot of mistakes on the test dataset.

**2. High variance:** As mentioned, a Decision Tree generally leads to the overfitting of data. Due to the overfitting, there is more likely a chance of high variance in the output which leads to many errors in the final predictions and shows high inaccuracy in the results. So, in order to achieve zero bias (overfitting), it leads to high variance due to the bias-variance tradeoff.

**3. Unstable:** When we add new data points it can lead to regeneration of the overall Tree. Therefore, all nodes need to be recalculated and reconstructed.

**4. Not suitable for large datasets:** If the data size is large, then one single Tree may grow complex and lead to overfitting. So in this case, we should use Random Forest instead, an ensemble technique of a single Decision Tree.



# Notes of today's Project.

## House Price Prediction Using Multiple Linear Regression

### Problem Statement

- Consider a real estate company that has a dataset containing the prices of properties in the Delhi region. It wishes to use the data to optimise the sale prices of the properties based on important factors such as area, bedrooms, parking, etc.

- To identify the variables affecting house prices, e.g. area, number of rooms, bathrooms, etc.

- To create a linear model that quantitatively relates house prices with variables such as number of rooms, area, number of bathrooms, etc.

- To know the accuracy of the model, i.e. how well these variables can predict house prices.

### Data set description

- Data is collected from https://www.kaggle.com/ashydv/housing-dataset .







## Discuss 'Naive' in a Naive Bayes algorithm?

- The Naive Bayes Algorithm model is based on the Bayes Theorem. It describes the probability of an event. It is based on prior knowledge of conditions which might be related to that specific event.

## **Why is Naive Bayes naive?**

- Naive Bayes is a machine learning implementation of [Bayes Theorem](http://machinelearningspecialist.com/machine-learning-interview-questions-q6-bayes-theorem/).  It is a classification algorithm that predicts the probability of each data point belonging to a class and then classifies the point as the class with the highest probability.

- It is naive because while it uses conditional probability to make classifications, the algorithm simply assumes that all features of a class are independent.  This is considered naive because, in reality, it is not often the case.  The upside is that the math is simpler, the classifier runs quicker, and the results are often quite good for certain problems.

## What is conditional probability?

- conditional probability is a measure of the probability of an event occurring, given that another event (by assumption, presumption, assertion or evidence) has already occurred. If the event of interest is *A* and the event *B* is known or assumed to have occurred, "the conditional probability of *A* given *B*", or "the probability of *A* under the condition *B*", is usually written as P(*A*|*B*),or sometimes P*B*(*A*) or P(*A*/*B*). For example, the probability that any given person has a cough on any given day may be only 5%. But if we know or assume that the person is sick, then they are much more likely to be coughing. For example, the conditional probability that someone unwell is coughing might be 75%, in which case we would have that P(Cough) = 5% and P(Cough|Sick) = 75%.

## What is Bayes’ Theorem?

- Bayes’ Theorem gives us the probability of an event actually happening by combining the conditional probability given some result and the prior knowledge of an event happening.

- Conditional probability is the probability that something will happen, given that something has a occurred.  In other words, the conditional probability is the probability of X given a test result or P(X|Test).  For example, what is the probability an e-mail is spam given that my spam filter classified it as spam.

- The prior probability is based on previous experience or the percentage of previous samples.  For example, what is the probability that any email is spam.

- *Formally*

![img](http://machinelearningspecialist.com/wp-content/uploads/2017/07/bayes-300x78.png)

- - P(A|B) = Posterior probability = Probability of A given B happened
  - P(B|A) = Conditional probability = Probability of B happening if A is true
  - P(A) = Prior probability = Probability of A happening in general
  - P(B) = Evidence probability = Probability of getting a positive test

## What are the types of Naive Bayes Classifier?

There are several types of Naive Bayes classifiers.  Which one you use will depend on the features you are working with. 

- The different types are:
  - Gaussian NB – use when you have continuous feature values.  This classifier assumes each class is normally distributed.
  - MultiNomial NB – good for text classification.  This classifier treats each occurrence of a word as an event.
  - Bernoulli NB – use when you have multiple features that are assumed to be binary.  This classifier can be used for text classification but the features must be binary.  For text classification, the features can be set as a word is in the document or not in the document.

## What are the advantages and dis-advantages of Naive Bayes Classification?

*Advantages*

- Can successfully train on small data set.
- Good for text classification, good for multi class classification.
- Quick and simple calculation since it is naive.

*Disadvantages*

- Can’t learn the relationship among the features because assumes feature independence.
- Continuous feature data is assumed to be normally distributed.



## Which language is best for text analytic? R or Python?

- Python will more suitable for text analytic as it consists of a rich library known as pandas. It allows you to use high-level data analysis tools and data structures, while R doesn't offer this feature.



## Explain the steps for a Data analytics project?

- The following are important steps involved in an analytics project:
  - Understand the Business problem
  - Explore the data and study it carefully.
  - Prepare the data for modelling by finding missing values and transforming variables.
  - Start running the model and analyse the Big data result.
  - Validate the model with new data set.
  - Implement the model and track the result to analyse the performance of the model for a specific period.




## List out the libraries in Python used for Data Analysis and Scientific Computations.

- SciPy
- Pandas
- Matplotlib
- NumPy
- SKLearn
- Seaborn







## What does SVM stands for?

- Support vector machines

## What is SVM?

- Support vector machines is a supervised machine learning algorithm which works both on classification and regression problems. It tries to classify data by finding a hyperplane that maximizes the margin between the classes in the training data. Hence, SVM is an example of a large margin classifier.
- The basic idea of support vector machines:
  - Optimal hyperplane for linearly separable patterns
  - Extend to patterns that are not linearly separable by transformations of original data to map into new space(i.e the kernel trick)

## Explain SVM to a non-technical person.

- Suppose you have to construct a bidirectional road. Now you have to make a dividing line. The optimal approach would be to make margins on the sides and draw an equidistant line from both the margins.
- This is exactly how SVM tries to classify points by finding an optimal centre line technically called as hyper plane.

## 3. What is the geometric intuition behind SVM?

- If you are asked to classify two different classes. There can be multiple hyperplanes which can be drawn.

  ![a.png](https://miro.medium.com/max/469/0*j6b6qNc-E0RfBxFj)

- SVM chooses the hyperplane which separates the data points as widely as possible. SVM draws a hyperplane parallel to the actual hyperplane intersecting with the first point of class A (also known as Support Vectors) and another hyperplane parallel to the actual hyperplane intersecting with the first point of class B. SVM tries to maximize these margins. Eventually, this margin maximization improves the model’s accuracy on unseen data.

##  How would explain Convex Hull in light of SVMs?

- We simply build a convex hull for class A and class B and draw a perpendicular on the shortest distance between the closest points of both these hulls.

  ![a.png](https://miro.medium.com/max/338/0*0f3JsP3NsoWhYwKs)

## What do know about Hard Margin SVM and Soft Margin SVM?

- Explanation: If a point Xi satisfies the equation ***Yi(WT\*Xi +b) ≥ 1,\*** then Xi is correctly classified else incorrectly classified. So we can see that if the points are linearly separable then only our hyperplane is able to distinguish between them and if any outlier is introduced then it is not able to separate them. So these type of SVM is called ***hard margin SVM\*** *(since we have very strict constraints to correctly classify each and every data point).*

- To overcome this, we introduce a term ***( ξ )\*** (pronounced as Zeta)

  ![a.png](https://miro.medium.com/max/183/0*Mimj5tNTj0RoCpRv)

  *if ξi= 0, the points can be considered as correctly classified.*

  *if ξi> 0 , Incorrectly classified points.*

## What is Hinge Loss?

- Hinge Loss is a loss function which penalises the SVM model for inaccurate predictions.

- If ***Yi(WT\*Xi +b) ≥ 1\***, hinge loss is ‘**0**’ i.e the points are correctly classified. When  ***Yi(WT\*Xi +b)\* < 1**, then hinge loss increases massively.

- As ***Yi(WT\*Xi +b)\*** increases with every misclassified point, the upper bound of hinge loss {**1- \*Yi(WT\*Xi +b)\***} also increases exponentially.

- Hence, the points that are farther away from the decision margins have a greater loss value, thus penalising those points.

  ![a.png](https://miro.medium.com/max/273/0*ZnoTexvPg-6hGTHB)

  We can formulate hinge loss as **max[0, 1- \*Yi(WT\*Xi +b)\*]**

## Explain the Dual form of SVM formulation?

- The aim of the Soft Margin formulation is to minimize

  ![a.png](https://miro.medium.com/max/131/0*ijBJ9cN63ydw0Nsv)

  subject to

  ![a.png](https://miro.medium.com/max/431/0*W1bCbvkRYDZjtoMU)

- This is also known as the primal form of SVM.

- The duality theory provides a convenient way to deal with the constraints. The dual optimization problem can be written in terms of dot products, thereby making it possible to use kernel functions.

- It is possible to express a different but closely related problem, called its dual problem. The solution to the dual problem typically gives a lower bound to the solution of the primal problem, but under some conditions, it can even have the same solutions as the primal problem. Luckily, the SVM problem happens to meet these conditions, so you can choose to solve the primal problem or the dual problem; both will have the same solution.

  ![a.png](https://miro.medium.com/max/1000/0*d7YVJ-Pb3dj2m-8f)

## What’s the “kernel trick” and how is it useful?

- Earlier we have discussed applying SVM on linearly separable data but it is very rare to get such data. Here, kernel trick plays a huge role. The idea is to map the non-linear separable data-set into a higher dimensional space where we can find a hyperplane that can separate the samples.

  ![a.png](https://miro.medium.com/max/838/0*ZnINGVLyQZfrcZYG)

- It reduces the complexity of finding the mapping function. So, **Kernel function defines the inner product in the transformed space.** Application of the kernel trick is not limited to the SVM algorithm. Any computations involving the dot products (x, y) can utilize the kernel trick.

## Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?**

- This question applies only to linear SVMs since kernelized can only use the dual form. The computational complexity of the primal form of the SVM problem is proportional to the number of training instances m, while the computational complexity of the dual form is proportional to a number between m² and m³. So, if there are millions of instances, you should use the primal form, because the dual form will be much too slow.

## Explain about SVM Regression?

- The Support Vector Regression (SVR) uses the same principles as the SVM for classification, with only a few minor differences. First of all, because the output is a real number it becomes very difficult to predict the information at hand, which has infinite possibilities. In the case of regression, a margin of tolerance (epsilon) is set in approximation to the SVM

  ![img](https://miro.medium.com/max/458/0*eXs0bEy10djpwu7d)

## Give some situations where you will use an SVM over a RandomForest Machine Learning algorithm.

  - The main reason to use an SVM instead is that the problem might not be linearly separable. In that case, we will have to use an SVM with a non-linear kernel (e.g. RBF).

  - Another related reason to use SVMs is if you are in a higher-dimensional space. For example, SVMs have been reported to work better for text classification.

## SVM being a large margin classifier, is it influenced by outliers?

- Yes, if C is large, otherwise not.

## In SVM, what is the angle between the decision boundary and theta?

- Decision boundary is a plane having equation Theta1*x1+Theta2*x2+……+c = 0, so as per the property of a plane, it’s coefficients vector is normal to the plane. Hence, the Theta vector is perpendicular to the decision boundary.

## Can we apply the kernel trick to logistic regression? Why is it not used in practice then?

    1. Logistic Regression is computationally more expensive than SVM — O(N³) vs O(N²k) where k is the number of support vectors.
    2. The classifier in SVM is designed such that it is defined only in terms of the support vectors, whereas in Logistic Regression, the classifier is defined over all the points and not just the support vectors. This allows SVMs to enjoy some natural speed-ups (in terms of efficient code-writing) that is hard to achieve for Logistic Regression.

## What is the difference between logistic regression and SVM without a kernel?

- They differ only in the implementation . SVM is much more efficient and has good optimization packages.



## Does SVM give any probabilistic output?

- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation









## What is bias?

- Bias is an error introduced in your model due to oversimplification of the machine learning algorithm. It can lead to underfitting. When you train your model at that time model makes simplified assumptions to make the target function easier to understand.
- Low bias machine learning algorithms — Decision Trees, k-NN and SVM High bias machine learning algorithms — Linear Regression, Logistic Regression

## What is Variance?

- Variance is error introduced in your model due to complex machine learning algorithm, your model learns noise also from the training data set and performs badly on test data set. It can lead to high sensitivity and overfitting.
- Normally, as you increase the complexity of your model, you will see a reduction in error due to lower bias in the model. However, this only happens until a particular point. As you continue to make your model more complex, you end up over-fitting your model and hence your model will start suffering from high variance.

## What is Selection Bias?

- Selection bias is a kind of error that occurs when the researcher decides who is going to be studied. It is usually associated with research where the selection of participants isn’t random. It is sometimes referred to as the selection effect. It is the distortion of statistical analysis, resulting from the method of collecting samples. If the selection bias is not taken into account, then some conclusions of the study may not be accurate.

## Types of selection bias?

- Sampling bias: It is a systematic error due to a non-random sample of a population causing some members of the population to be less likely to be included than others resulting in a biased sample.

- Time interval: A trial may be terminated early at an extreme value (often for ethical reasons), but the extreme value is likely to be reached by the variable with the largest variance, even if all variables have a similar mean.

- Data: When specific subsets of data are chosen to support a conclusion or rejection of bad data on arbitrary grounds, instead of according to previously stated or generally agreed criteria.

- Attrition: Attrition bias is a kind of selection bias caused by attrition (loss of participants) discounting trial subjects/tests that did not run to completion.

## What is bias-variance trade-off?

![a.png](https://miro.medium.com/max/547/0*goZnsEUDxSoE3v3N.png)

- Bias-Variance trade-off: The goal of any supervised machine learning algorithm is to have low bias and low variance to achieve good prediction performance.

- The k-nearest neighbour algorithm has low bias and high variance, but the trade-off can be changed by increasing the value of k which increases the number of neighbours that contribute to the prediction and in turn increases the bias of the model.

- The support vector machine algorithm has low bias and high variance, but the trade-off can be changed by increasing the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.

- There is no escaping the relationship between bias and variance in machine learning. Increasing the bias will decrease the variance. Increasing the variance will decrease bias.
  .

## What is a confusion matrix?

![a.png](https://miro.medium.com/max/332/0*n0yJZxhcrt6LuHBV.png)

- The confusion matrix is a 2X2 table that contains 4 outputs provided by the binary classifier. Various measures, such as error-rate, accuracy, specificity, sensitivity, precision and recall are derived from it. Confusion Matrix
  A data set used for performance evaluation is called a test data set. It should contain the correct labels and predicted labels.

![a.png](https://miro.medium.com/max/400/0*x5BeBND2si5OsgZf.png)

#### The predicted labels will exactly the same if the performance of a binary classifier is perfect.

![a.png](https://miro.medium.com/max/480/0*nvaqNjPZTCtUG0uj.png)

#### The predicted labels usually match with part of the observed labels in real-world scenarios.

![a.png](https://miro.medium.com/max/480/0*jFrvfY-PMDbG3CeR.png)

#### A binary classifier predicts all data instances of a test data set as either positive or negative. This produces four outcomes-

True-positive(TP) — Correct positive prediction

False-positive(FP) — Incorrect positive prediction

True-negative(TN) — Correct negative prediction

False-negative(FN) — Incorrect negative prediction

![a.png](https://miro.medium.com/max/480/0*XBpeI_iDvFkSbJCT.png)

#### Basic measures derived from the confusion matrix

Error Rate = (FP+FN)/(P+N)

Accuracy = (TP+TN)/(P+N)

Sensitivity(Recall or True positive rate) = TP/P

Specificity(True negative rate) = TN/N

Precision(Positive predicted value) = TP/(TP+FP)

F-Score(Harmonic mean of precision and recall) = (1+b)(PREC.REC)/(b²PREC+REC) where b is commonly 0.5, 1, 2.

 









## What is Ensemble Learning. Give an important example of Ensemble Learning?

- Ensemble Learning is a process of accumulating multiple models to form a better prediction model. In Ensemble Learning the performance of the individual model contributes to the overall development in every step. There are two common techniques in this – Bagging and Boosting.
- The most important example of Ensemble Learning is Random Forest Classifier. It takes multiple Decision Tree combined to form a better performance Random Forest model.



![img](https://miro.medium.com/max/2000/1*zTgGBTQIMlASWm5QuS2UpA.jpeg)

## What is Bagging?

-  In this the data set is split to perform parallel processing of models and results are accumulated based on performance to achieve better accuracy.
-  Bagging method helps you to implement similar learners on small sample populations. It helps you to make nearer predictions.

## What is Boosting?

- This is a sequential technique in which a result from one model is passed to another model to reduce error at every step making it a better performance model.
- Boosting is an iterative method which allows you to adjust the weight of an observation depends upon the last classification. Boosting decreases the bias error and helps you to build strong predictive models.



## What is Imbalanced Data? How do you manage to balance the data?

- If a data is distributed across different categories and the distribution is highly imbalance. Such data are known as Imbalance Data. These kind of datasets causes error in model performance by making category with large values significant for the model resulting in an inaccurate model.

- There are various techniques to handle imbalance data. We can increase the number of samples for minority classes. We can decrease the number of samples for classes with extremely high numbers of data points. We can use a cluster based technique to increase number of Data points for all the categories.

## **Explain Unsupervised Clustering approach?**

Answer : Grouping the data into different clusters based on the distribution of data is known as Clustering technique.

There are various Clustering Techniques –

- Density Based Clustering – DBSCAN , HDBSCAN

- Hierarchical Clustering.

- Partition Based Clustering

- Distribution Based Clustering.

## **Explain Recommender Systems?**

- The goal of a recommender system is to generate meaningful recommendations to a collection of users for items or products that might interest them. 
- Suggestions for books on Amazon, or movies on Netflix, are real-world examples of the operation of industry-strength recommender systems. 
- The design of such recommendation engines depends on the domain and the particular characteristics of the data available. For example, movie watchers on Netflix frequently provide ratings on a scale of 1 (disliked) to 5 (liked). 
- Such a data source records the quality of interactions between users and items. Additionally, the system may have access to user-specific and item-specific profile attributes such as demographics and product descriptions, respectively. 









## What is time series data?

- A **time series** is a **data** set that tracks a sample over **time**. In particular, a **time series** allows one to see what factors influence certain variables from period to period. **Time series analysis** can be useful to see how a given asset, security, or economic variable changes over **time**.

## What is time series forecasting?

- **Time series forecasting** uses information regarding historical values and associated patterns to predict future activity. Most often, this relates to trend analysis, cyclical fluctuation analysis, and issues of seasonality.



##  Give some example of time series problem?

**1. Estimating number of hotel rooms booking in next 6 months.**
**2. Estimating the total sales in next 3 years of an insurance company.**
**3. Estimating the number of calls for the next one week.**

A) Only 3
B) 1 and 2
C) 2 and 3
D) 1 and 3
E) 1,2 and 3

Solution: **(E)**

All the above options have a time component associated.

 

## Give some example of a time series model?

A) Naive approach
B) Exponential smoothing
C) Moving Average

Naïve approach: Estimating technique in which the last period’s actuals are used as this period’s forecast, without adjusting them or attempting to establish causal factors. It is used only for comparison with the forecasts generated by the better (sophisticated) techniques.

In exponential smoothing, older data is given progressively-less relative importance whereas newer data is given progressively-greater importance.

In time series analysis, the moving-average (MA) model is a common approach for modeling univariate time series. The moving-average model specifies that the output variable depends linearly on the current and various past values of a stochastic (imperfectly predictable) term.

 

## What are the component for a time series plot?

A) Seasonality
B) Trend
C) Cyclical
D) Noise

A seasonal pattern exists when a series is influenced byseasonal factors (e.g., the quarter of the year, the month, or day of the week). Seasonality is always of a fixed and known period. Hence, seasonal time series are sometimes called periodic time series

Seasonality is always of a fixed and known period. A cyclic pattern exists when data exhibit rises and falls that are not of fixed period.

Trend is defined as the ‘long term’ movement in a time series without calendar related and irregular effects, and is a reflection of the underlying level. It is the result of influences such as population growth, price inflation and general economic changes. The following graph depicts a series in which there is an obvious upward trend over time.

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10043145/TS1.gif)

Quarterly Gross Domestic Product

Noise: In discrete time, white noise is a discrete signal whose samples are regarded as a sequence of serially uncorrelated random variables with zero mean and finite variance.

 

 

## Smoothing parameter close to one gives more weight or influence to recent observations over the forecast. Why?

It may be sensible to attach larger weights to more recent observations than to observations from the distant past. This is exactly the concept behind simple exponential smoothing. Forecasts are calculated using weighted averages where the weights decrease exponentially as observations come from further in the past — the smallest weights are associated with the oldest observations:

y^T+1|T=αyT+α(1−α)yT−1+α(1−α)2yT−2+⋯,(7.1)

where 0≤α≤10≤α≤1 is the smoothing parameter. The one-step-ahead forecast for time T+1T+1 is a weighted average of all the observations in the series y1,…,yT. The rate at which the weights decrease is controlled by the parameter αα.

 

## Sum of weights in exponential smoothing is _____

 1

Table 7.1 shows the weights attached to observations for four different values of αα when forecasting using simple exponential smoothing. Note that the sum of the weights even for a small αα will be approximately one for any reasonable sample size.

| **Observation** | α=0.2      | α=0.4      | α=0.6      | α=0.8      |
| --------------- | ---------- | ---------- | ---------- | ---------- |
| yT              | 0.2        | 0.4        | 0.6        | 0.8        |
| yT−1            | 0.16       | 0.24       | 0.24       | 0.16       |
| yT−2            | 0.128      | 0.144      | 0.096      | 0.032      |
| yT−3            | 0.102      | 0.0864     | 0.0384     | 0.0064     |
| yT−4            | (0.2)(0.8) | (0.4)(0.6) | (0.6)(0.4) | (0.8)(0.2) |
| yT−5            | (0.2)(0.8) | (0.4)(0.6) | (0.6)(0.4) | (0.8)(0.2) |

 

## The last period’s forecast was 70 and demand was 60. What is the simple exponential smoothing forecast with alpha of 0.4 for the next period

A) 63.8
B) 65
C) 62
D) 66

Solution: **(D)**

Yt-1= 70

St-1= 60

Alpha = 0.4

Substituting the values we get

0.4*60 + 0.6*70= 24 + 42= 66

 

## What does autocovariance measures?

A) Linear dependence between multiple points on the different series observed at different times
B)Quadratic dependence between two points on the same series observed at different times
C) Linear dependence between two points on different series observed at same time
D) Linear dependence between two points on the same series observed at different times

Solution: **(D)**

Option D is the definition of autocovariance.

## Which of the following is not a necessary condition for weakly stationary time series

A) Mean is constant and does not depend on time
B) Autocovariance function depends on s and t only through their difference |s-t| (where t and s are moments in time)
C) The time series under considerations is a finite variance process
D) Time series is Gaussian

Solution: **(D)**

A Gaussian time series implies stationarity is strict stationarity.

## Which of the following is not a technique used in smoothing time series?

A) Nearest Neighbour Regression
B)  Locally weighted scatter plot smoothing
C) Tree based models like (CART)
D) Smoothing Splines

Solution: **(C)**

Time series smoothing and ﬁltering can be expressed in terms of local regression models. Polynomials and regression splines also provide important techniques for smoothing. CART based models do not provide an equation to superimpose on time series and thus cannot be used for smoothing. All the other techniques are well documented smoothing techniques.

 

## If the demand is 100 during October 2016, 200 in November 2016, 300 in December 2016, 400 in January 2017. What is the 3-month simple moving average for February 2017

A) 300
B) 350
C) 400
D) Need more information

Solution: **(A)**

X`= (xt-3 + xt-2 + xt-1 ) /3

(200+300+400)/ 3 = 900/3 =300

 

## Looking at the below ACF plot, would you suggest to apply AR or MA in ARIMA modeling technique

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10054154/Image141.jpg)
A) AR
B) MA
C) Can’t Say

Solution: **(A)**

MA model is considered in the following situation, If the autocorrelation function (ACF) of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative–i.e., if the series appears slightly “overdifferenced”–then consider adding an MA term to the model. The lag beyond which the ACF cuts off is the indicated number of MA terms.

But as there are no observable sharp cutoffs the AR model must be preffered.

 

## Suppose, you are a data scientist at Analytics Vidhya. And you observed the views on the articles increases during the month of Jan-Mar. Whereas the views during Nov-Dec decreases

**Does the above statement represent seasonality?**

A) TRUE
B) FALSE
C) Can’t Say

Solution: **(A)**

Yes this is a definite seasonal trend as there is a change in the views at particular times.

Remember, Seasonality is a presence of variations at specific periodic intervals.

 

## Which of the following graph can be used to detect seasonality in time series data

**1. Multiple box**
**2. Autocorrelation**

A) Only 1
B) Only 2
C) 1 and 2
D)  None of these

Solution: **(C)**

Seasonality is a presence of variations at specific periodic intervals.

The variation of distribution can be observed in multiple box plots. And thus seasonality can be easily spotted. *Autocorrelation plot* should show spikes at lags equal to the period.

 

## Stationarity is a desirable property for a time series process

A) TRUE
B) FALSE

Solution: **(A)**

When the following conditions are satisfied then a time series is stationary.

1. Mean is constant and does not depend on time
2. Autocovariance function depends on s and t only through their difference |s-t| (where t and s are moments in time)
3. The time series under considerations is a finite variance process

These conditions are essential prerequisites for mathematically representing a time series to be used for analysis and forecasting. Thus stationarity is a desirable property.

 

## Suppose you are given a time series dataset which has only 4 columns (id, Time, X, Target)

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10055051/TS3.png)**What would be the rolling mean of feature X if you are given the window size 2?**

**Note: X column represents rolling mean.**

A) ![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10055104/TSO1.png)

B) ![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10055116/TSO2.png)

C) ![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10055126/TSO3.png)

D) None of the above

Solution: **(B)**

X`= xt-2 + xt-1 /2

Based on the above formula: (100 +200) /2 =150; (200+300)/2 = 250 and so on.

 

## Imagine, you are working on a time series dataset. Your manager has asked you to build a highly accurate model. You started to build two types of models which are given below

**Model 1: Decision Tree model**

**Model 2: Time series regression model**

**At the end of evaluation of these two models, you found that model 2 is better than model 1. What could be the possible reason for your inference?**

A) Model 1 couldn’t map the linear relationship as good as Model 2
B) Model 1 will always be better than Model 2
C) You can’t compare decision tree with time series regression
D) None of these

Solution: **(A)**

A time series model is similar to a regression model. So it is good at finding simple linear relationships. While a tree based model though efficient will not be as good at finding and exploiting linear relationships.

 

## What type of analysis could be most effective for predicting temperature on the following type of data

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10055713/ts.jpg)
A) Time Series Analysis
B) Classification
C)  Clustering
D) None of the above

Solution: **(A)**

The data is obtained on consecutive days and thus the most effective type of analysis will be time series analysis.

 

## What is the first difference of temperature / precipitation variable

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10060016/TS4.jpg)
A) 15,12.2,-43.2,-23.2,14.3,-7
B) 38.17,-46.11,-4.98,14.29,-22.61
C) 35,38.17,-46.11,-4.98,14.29,-22.61
D) 36.21,-43.23,-5.43,17.44,-22.61

Solution: **(B)**

73.17-35 = 38.17

27.05-73.17 = – 46.11 and so on..

13.75 – 36.36 = -22.61

## Consider the following set of data:

**{23.32 32.33 32.88 28.98 33.16 26.33 29.88 32.69 18.98 21.23 26.66 29.89}**

**What is the lag-one sample autocorrelation of the time series?**

A) 0.26
B) 0.52
C) 0.13
D) 0.07

Solution: **(C)**

ρˆ1 = PT t=2(xt−1−x¯)(xt−x¯) PT t=1(xt−x¯) 2

= (23.32−x¯)(32.33−x¯)+(32.33−x¯)(32.88−x¯)+··· PT t=1(xt−x¯) 2

= 0.130394786

Where x¯ is the mean of the series which is 28.0275

 

## Any stationary time series can be approximately the random superposition of sines and cosines oscillating at various frequencies

A) TRUE
B) FALSE

Solution: **(A)**

A weakly stationary time series, xt, is a ﬁnite variance process such that

- The mean value function, µt, is constant and does not depend on time t, and (ii) the autocovariance function, γ(s,t), deﬁned in depends on s and t only through their diﬀerence |s−t|.

random superposition of sines and cosines oscillating at various frequencies is white noise. white noise is weakly stationary or stationary. If the white noise variates are also normally distributed or Gaussian, the series is also strictly stationary.

 

## Autocovariance function for weakly stationary time series does not depend on _______ ?

A)  Separation of xs and xt
B) h = | s – t |
C) Location of point at a particular time

Solution: **(C)**

By definition of weak stationary time series described in previous question.

## Two time series are jointly stationary if _____ 

A) They are each stationary
B) Cross variance function is a function only of lag h

A) Only A
B) Both A and B

Solution: **(D)**

Joint stationarity is defined based on the above two mentioned conditions.

 

## In autoregressive models _______ 

A) Current value of dependent variable is influenced by current values of independent variables
B) Current value of dependent variable is influenced by current and past values of independent variables
C) Current value of dependent variable is influenced by past values of both dependent and independent variables
D) None of the above

Solution: **(C)**

Autoregressive models are based on the idea that the current value of the series, xt, can be explained as a function of p past values, xt−1,xt−2,…,xt−p, where p determines the number of steps into the past needed to forecast the current value. Ex. xt = xt−1 −.90xt−2 + wt,

Where xt-1 and xt-2 are past values of dependent variable and wt the white noise can represent values of independent values.

The example can be extended to include multiple series analogous to multivariate linear regression.

 

## For MA (Moving Average) models the pair σ = 1 and θ = 5 yields the same autocovariance function as the pair σ = 25 and θ = 1/5![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10061249/TS5.jpg)![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10061259/TS6.jpg)

A) TRUE
B) FALSE

Solution: **(A)**

True, because autocovariance is invertible for MA models

note that for an MA(1) model, ρ(h) is the same for θ and 1 /θ

try 5 and 1 5, for example. In addition, the pair σ2 w = 1 and θ = 5 yield the same autocovariance function as the pair σ2 w = 25 and θ = 1/5.

 

## How many AR and MA terms should be included for the time series by looking at the above ACF and PACF plots

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10061918/Capture23.png)

A) AR (1) MA(0)
B) AR(0)MA(1)
C) AR(2)MA(1)
D) AR(1)MA(2)
E) Can’t Say

Solution: **(B)**

Strong negative correlation at lag 1 suggest MA and there is only 1 significant lag. Read [this article ](https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/)for a better understanding.

 

## Which of the following is true for white noise?

A) Mean =0
B) Zero autocovariances
C) Zero autocovariances except at lag zero
D) Quadratic Variance

Solution: **(C)**

A white noise process must have a constant mean, a constant variance and no autocovariance structure (except at lag zero, which is the variance).

 

## For the following MA (3) process** ***yt\* = \*μ\* + \*Εt\* + \*θ\*1\*Εt\*-1 + \*θ\*2\*Εt\*-2 + \*θ\*3\*Εt\*-3 , where \*σt\* is a zero mean white noise process with variance \*σ\*

A) ACF = 0 at lag 3
B) ACF =0 at lag 5
C) ACF =1 at lag 1
D) ACF =0 at lag 2
E) ACF = 0 at lag 3 and at lag 5

Solution: **(B)**

Recall that an MA(q) process only has memory of length q. This means that all of the autocorrelation coefficients will have a value of zero beyond lag q. This can be seen by examining the MA equation, and seeing that only the past q disturbance terms enter into the equation, so that if we iterate this equation forward through time by more than q periods, the current value of the disturbance term will no longer affect y. Finally, since the autocorrelation function at lag zero is the correlation of y at time t with y at time t (i.e. the correlation of y_t with itself), it must be one by definition.

 

## Consider the following AR(1) model with the disturbances having zero mean and unit variance

***yt\*= 0.4 + 0.2\*yt\*-1+\*ut\***

**The (unconditional) variance of y will be given by ?**

A)  1.5
B) 1.04
C) 0.5
D)  2

Solution: **(B)**

Variance of the disturbances divided by (1 minus the square of the autoregressive coefficient

Which in this case is : 1/(1-(0.2^2))= 1/0.96= 1.041

 

## The pacf (partial autocorrelation function) is necessary for distinguishing between ______ 

A) An AR and MA model is_solution: False
B)  An AR and an ARMA is_solution: True
C) An MA and an ARMA is_solution: False
D) Different models from within the ARMA family

Solution: **(B)**

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10062448/TS7.png)

 

## Second differencing in time series can help to eliminate which trend

A) Quadratic Trend
B) Linear Trend
C) Both A & B
D) None of the above

Solution: (A)

The ﬁrst diﬀerence is denoted as ∇xt = xt −xt−1. (1)

As we have seen, the ﬁrst diﬀerence eliminates a linear trend. A second diﬀerence, that is, the diﬀerence of (1), can eliminate a quadratic trend, and so on.

 

## Which of the following cross validation techniques is better suited for time series data

A)  k-Fold Cross Validation
B) Leave-one-out Cross Validation
C) Stratified Shuffle Split Cross Validation
D) Forward Chaining Cross Validation

Solution: **(D)**

Time series is ordered data. So the validation data must be ordered to. Forward chaining ensures this. It works as follows:

- fold 1 : training [1], test [2]
- fold 2 : training [1 2], test [3]
- fold 3 : training [1 2 3], test [4]
- fold 4 : training [1 2 3 4], test [5]
- fold 5 : training [1 2 3 4 5], test [6]

## BIC penalizes complex models more strongly than the AIC.

A) TRUE
B) FALSE

Solution: **(A)**

AIC = -2*ln(likelihood) + 2*k,

BIC = -2*ln(likelihood) + ln(N)*k,

where:

k = model degrees of freedom

N = number of observations

At relatively low N (7 and less) BIC is more tolerant of free parameters than AIC, but less tolerant at higher N (as the natural log of N overcomes 2).

 

## The figure below shows the estimated autocorrelation and partial autocorrelations of a time series of n = 60 observations. Based on these plots, we should.

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10062952/TS8.png)

A) Transform the data by taking logs
B) Difference the series to obtain stationary data
C) Fit an MA(1) model to the time series

Solution: **(B)**

The autocorr shows a definite trend and partial autocorrelation shows a choppy trend, in such a scenario taking a log would be of no use. Differencing the series to obtain a stationary series is the only option.

 

**Question Context (37-38)**

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/04/10063336/TS10.jpg)

## Use the estimated exponential smoothening given above and predict temperature for the next 3 years (1998-2000

**These results summarize the fit of a simple exponential smooth to the time series**.

A) 0.2,0.32,0.6
B) 0.33, 0.33,0.33
C) 0.27,0.27,0.27
D) 0.4,0.3,0.37

Solution: **(B)**

The predicted value from the exponential smooth is the same for all 3 years, so all we need is the value for next year. The expression for the smooth is

smootht = α yt + (1 – α) smooth t-1

Hence, for the next point, the next value of the smooth (the prediction for the next observation) is

smoothn = α yn + (1 – α) smooth n-1

= 0.3968*0.43 + (1 – 0.3968)* 0.3968

= 0.3297

 

## Find 95% prediction intervals for the predictions of temperature in 1999.

**These results summarize the fit of a simple exponential smooth to the time series.**

A) 0.3297 2 * 0.1125
B) 0.3297 2 * 0.121
C) 0.3297 2 * 0.129
D) 0.3297 2 * 0.22

Solution: **(B)**

The sd of the prediction errors is

1 period out 0.1125

2 periods out 0.1125 sqrt(1+α2) = 0.1125 * sqrt(1+ 0.39682) ≈ 0.121

 

## Which of the following statement is correct

**1. If autoregressive parameter (p) in an ARIMA model is 1, it means that there is no auto-correlation in the series.**
**2. If moving average component (q) in an ARIMA model is 1, it means that there is auto-correlation in the series with lag 1.**
**3. If integrated component (d) in an ARIMA model is 0, it means that the series is not stationary.**

A) Only 1
B) Both 1 and 2
C) Only 2
D)  All of the statements

Solution: **(C)**

Autoregressive component: AR stands for autoregressive. Autoregressive parameter is denoted by p. When p =0, it means that there is no auto-correlation in the series. When p=1, it means that the series auto-correlation is till one lag.

Integrated: In ARIMA time series analysis, integrated is denoted by d. Integration is the inverse of differencing. When d=0, it means the series is stationary and we do not need to take the difference of it. When d=1, it means that the series is not stationary and to make it stationary, we need to take the first difference. When d=2, it means that the series has been differenced twice. Usually, more than two time difference is not reliable.

Moving average component: MA stands for moving the average, which is denoted by q. In ARIMA, moving average q=1 means that it is an error term and there is auto-correlation with one lag.

## In a time-series forecasting problem, if the seasonal indices for quarters 1, 2, and 3 are 0.80, 0.90, and 0.95 respectively. What can you say about the seasonal index of quarter 4

A) It will be less than 1
B)  It will be greater than 1
C) It will be equal to 1
D) Seasonality does not exist
E) Data is insufficient

Solution: **(B)**

The seasonal indices must sum to 4, since there are 4 quarters. .80 + .90 + .95 = 2.65, so the seasonal index for the 4th quarter must be 1.35 so B is the correct answer.









## What is the Difference Between Supervised and Unsupervised Machine Learning?

- **Supervised learning -** This model learns from the labelled data and makes a future prediction as output 
- **Unsupervised learning -** This model uses unlabelled input data and allows the algorithm to act on that information without guidance.





## What is the Difference Between Inductive Machine Learning and Deductive Machine Learning? 

|                    **Inductive Learning**                    |                    **Deductive Learning**                    |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| It observes instances based on defined principles to draw a conclusion Example: Explaining to a child to keep away from the fire by showing a video where fire causes damage | It concludes experiences Example: Allow the child to play with fire. If he or she gets burned, they will learn that it is dangerous and will refrain from making the same mistake again |





## What is Unsupervised Learning?

Unsupervised learning is also a type of machine learning algorithm used to find patterns on the set of data given. In this, we don’t have any dependent variable or label to predict. Unsupervised Learning Algorithms:

- Clustering, 
- Anomaly Detection, 
- Neural Networks and Latent Variable Models.

**Example:**

In the same example, a T-shirt clustering will categorize as “collar style and V neck style”, “crew neck style” and “sleeve types”.





## What Are Unsupervised Machine Learning Techniques? 

There are two techniques used in unsupervised learning: clustering and association.



#### Clustering

Clustering problems involve data to be divided into subsets. These subsets, also called clusters, contain data that are similar to each other. Different clusters reveal different details about the objects, unlike classification or regression.

![Clustering](https://www.simplilearn.com/ice9/frs_images/1-clustering_subsets.jpg)



#### Association

In an association problem, we identify patterns of associations between different variables or items.

For example, an e-commerce website can suggest other items for you to buy, based on the prior purchases that you have made, spending habits, items in your wishlist, other customers’ purchase habits, and so on.

![Association](https://www.simplilearn.com/ice9/frs_images/1-association_subset.jpg)







## **Explain what is the function of ‘Unsupervised Learning’?**

- Find clusters of the data
- Find low-dimensional representations of the data
- Find interesting directions in data
- Interesting coordinates and correlations
- Find novel observations/ database cleaning





## What is Clustering?

Clustering is the process of grouping a set of objects into a number of groups. Objects should be similar to one another within the same cluster and dissimilar to those in other clusters.

A few types of clustering are:

- Hierarchical clustering
- K means clustering
- Density-based clustering
- Fuzzy clustering, etc.





## Compare K-means and KNN Algorithms.

|                         **K-means**                          |                             KNN                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| It is unsupervised K-Means is a clustering algorithm The points in each cluster are similar to each other, and each cluster is different from its neighbouring clusters | It is a  supervised in nature KNN is a classification algorithm It classifies an unlabelled observation based on its K (can be any number) surrounding neighbours |





## How can you select K for K-means Clustering?

There are two kinds of methods that include direct methods and statistical testing methods:

- Direct methods: It contains elbow and silhouette 
- Statistical testing methods: It has gap statistics.

The silhouette is the most frequently used while determining the optimal value of k.





## How is KNN different from k-means clustering?

**Answer:** K-Nearest Neighbors is a supervised classification algorithm, while k-means clustering is an unsupervised clustering algorithm. While the mechanisms may seem similar at first, what this really means is that in order for K-Nearest Neighbors to work, you need labeled data you want to classify an unlabeled point into (thus the nearest neighbor part). K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points.

The critical difference here is that KNN needs labeled points and is thus supervised learning, while k-means doesn’t—and is thus unsupervised learning.



## **What is K means Clustering Algorithm?**

K Means algorithm is a centroid-based clustering (unsupervised) technique. This technique groups the dataset into k different clusters having an almost equal number of points. Each of the clusters has a centroid point which represents the mean of the data points lying in that cluster.

The idea of the K-Means algorithm is to find k-centroid points and every point in the dataset will belong to either of the k-sets having minimum Euclidean distance.

## ** What is Lloyd’s algorithm for Clustering?**

It is an approximation iterative algorithm that is used to cluster the data points.

The steps of this algorithm are as follows:

- Initialization
- Assignment
- Update Centroid
- Repeat Steps 2 and 3 until convergence

### **Step-1: Initialization**

Randomly initialized k-centroids from the data points.

### **Step-2: Assignment**

For each observation in the dataset, calculate the euclidean distance between the point and all centroids. Then, assign a particular observation to the cluster with the nearest centroid.

### **Step-3: Updation of Centroid**

Now, observations in the clusters are changed. Therefore, update the value of the centroid with the new mean(average of all observations)value.

### **Step-4: Repeat Steps 2 and 3 until convergence**

Repeat steps 2 and 3 until the algorithm converges. If convergence is achieved then break the loop. Convergence refers to the condition where the previous value of centroids is equal to the updated value after the algorithm run.

## ** Is Feature Scaling required for the K means Algorithm?**

**Yes,** K-Means typically needs to have some form of normalization done on the datasets to work properly since it is sensitive to both the mean and variance of the datasets.

For performing feature scaling, generally, **StandardScaler** is recommended, but depending on the specific use cases, other techniques might be more suitable as well.

**For Example,** let’s have 2 variables, named age and salary where age is in the range of 20 to 60 and salary is in the range of 100-150K, since scales of these variables are different so when these variables are substituted in the euclidean distance formula, then the variable which is on the large scale suppresses the variable which is on the smaller scale. So, the impact of age will not be captured very clearly. Hence, you have to scale the variables to the same range using **Standard Scaler, Min-Max Scaler**, etc.

 

## ** Why do you prefer Euclidean distance over Manhattan distance in the K means Algorithm?**

Euclidean distance is preferred over Manhattan distance since Manhattan distance calculates distance only vertically or horizontally due to which it has dimension restrictions.

On the contrary, Euclidean distance can be used in any space to calculate the distances between the data points. Since in K means algorithm the data points can be present in any dimension, so Euclidean distance is a more suitable option.

## ** Why is the plot of the within-cluster sum of squares error (inertia) vs K in K means clustering algorithm elbow-shaped? Discuss if there exists any other possibility for the same with proper explanation.**

Let’s understand this with an example,

Say, we have **10 different data points** present, now consider the different cases:

- **k=10:** For the max value of k, all points behave as one cluster. So, within the cluster sum of squares is zero since only one data point is present in each of the clusters. So, at the max value of k, this should tend to zero.
- **K=1:** For the minimum value of k i.e, k=1, all these data points are present in the one cluster, and due to more points in the same cluster gives more variance i.e, more within-cluster sum of squares.
- **Between K=1 from K=10:** When you increase the value of k from 1 to 10, more points will go to other clusters, and hence the total within the cluster sum of squares (inertia) will come down. So, mostly this forms an elbow curve instead of other complex curves.

Hence, we can conclude that there does not exist any other possibility for the plot.

## ** Which metrics can you use to find the accuracy of the K means Algorithm?**

There does not exist a correct answer to this question as k means being an unsupervised learning technique does not discuss anything about the output column. As a result, one can not get the accuracy number or values from the algorithm directly.

## ** What is a centroid point in K means Clustering?**

Centroid point is the point that acts as a representative of a particular cluster and is the average of all the data points in the cluster which changes in each step (until convergence). Centroid can be calculated using the given formula:

![k means centroid](https://lh3.googleusercontent.com/GfGmyDrlthSCh3R0Bhr0b-jwVxGuqiMzagQ-G5tVE9RpE1rVQyPAum9Mbj9HKjlyOMzjBA1T2ila1G3GbsPMuEAPzWSXl7HpuP5G1nhudsLznbp6eFtoLNQgvunRSdLpeJPkW7jt)

**where,**

**Ci:** ith Centroid

**Si:** All points belonging to set-i with centroid as Ci

**xj:** jth point from the set

**||Si||:** number of points in set-i

## ** Does centroid initialization affect K means Algorithm?**

Yes, the final results of the k means algorithm depend on the centroid initialization as poor initialization can cause the algorithm to get stuck into an inferior local minimum.

## ** Discuss the optimization function for the K means Algorithm.**

![k means optimization functiion](https://lh4.googleusercontent.com/puGstYxqzBoRS6K0CGyOl3wDuzS9r71qD7O2-ctTFe7IZkVUKTpr_Z4qBRObSjk9eBcP6HyaP5TF_oazzfs-9uRzU6SyXKkVmrONPKqQWCs0Tehr1lmfqrP2HbLYd-d1GS0qIOn6)**
**

The objective of the K-Means algorithm is to find the k (k=no of clusters) number of centroids from C1, C2,——, Ck which minimizes the within-cluster sum of squares i.e, the total sum over each cluster of the sum of the square of the distance between the point and its centroid.

This cost comes under the **NP-hard problem** and therefore has **exponential time complexity**. So we come up with the idea of approximation using Lloyd’s Algorithm.

## ** What are the advantages and disadvantages of the K means Algorithm?**

### **Advantages:**

- Easy to understand and implement.
- Computationally efficient for both training and prediction.
- Guaranteed convergence.

### **Disadvantages:**

- We need to provide the number of clusters as an input variable to the algorithm.
- It is very sensitive to the initialization process.
- Good at clustering when we are dealing with spherical cluster shapes, but it will perform poorly when dealing with more complicated shapes.
- Due to the leveraging of the euclidean distance function, it is sensitive to outliers.

## ** What are the challenges associated with K means Clustering?**

The major challenge associated with k means clustering is its **initialization sensitivity**.

While finding the initial centroids for K-Means clustering using Lloyd’s algorithm, we were using randomization i.e, initial k-centroids were picked randomly from the data points.

This Randomization in picking the k-centroids creates the problem of initialization sensitivity which tends to affect the final formed clusters. As a result, the final formed clusters depend on how initial centroids were picked.

## ** What are the ways to avoid the problem of initialization sensitivity in the K means Algorithm?**

There are two ways to avoid the problem of initialization sensitivity:

- **Repeat K means:** It basically repeats the algorithm again and again along with initializing the centroids followed by picking up the cluster which results in the small intracluster distance and large intercluster distance.
- **K Means++:** It is a smart centroid initialization technique.

Amongst the above two techniques, K-Means++ is the best approach.

## ** What is the difference between K means and K means++ Clustering?**

In k-means, we randomly initialized the k number of centroids while in the k-means++ algorithm, firstly we initialized 1 centroid and for other centroids, we have to ensure that the next centroids are very far from the initial centroids which result in a lower possibility of the centroid being poorly initialized. As a result, the convergence is faster in K means++ clustering.

Moreover, in order to implement the k-means++ clustering using the **Scikit-learn** library, we set the parameters to **init = kmeans++** instead of **random**.

## ** How K means++ clustering Algorithm works?**

K Means++ algorithm is a smart technique for centroid initialization that initialized one centroid while ensuring the others to be far away from the chosen one resulting in faster convergence.

The steps to follow for centroid initialization are:

**Step-1:** Pick the first centroid point randomly.

**Step-2:** Compute the distance of all points in the dataset from the selected centroid. The distance of xi point from the farthest centroid can be calculated by the given formula:

![distance calculation](https://lh5.googleusercontent.com/fXNakdyi6ppb2yW7gs5j70gLeQK0RAu2gBIddgsjwigEGVL600gasaYBjT3DZ6blOfEpF2INRwloErHG7HQlnfUge2WyDL7MmSSXUqf96085wpoP_cMVbCbCuWRNVLxQdk9r3FKz)

**where,**

**di:** Distance of xi point from the farthest centroid

**m:** number of centroids already picked

**Step-3:** Make the point xi as the new centroid that is having maximum probability proportional to di

**Step-4:** Repeat the above last two steps till you find k centroids.

## ** How to decide the optimal number of K in the K means Algorithm?**

Most of the people give answers to this question directly as the Elbow Method however the explanation is only partially correct.

In order to find the optimal value of k, we need to observe our business problem carefully, along with analyzing the business inputs as well as the person who works on that data so that a decent idea regarding the optimal number of clusters can be extracted.

**For Example,** If we consider the data of a shopkeeper selling a product in which he will observe that some people buy things in summer, some in winter while some in between these two. So, the shopkeeper divides the customers into three categories. Therefore, K=3.

In cases where we do not get inference from the data directly we often use the following mentioned techniques:

- **Elbow Method –** This method finds the point of inflection on a graph of the percentage of variance explained to the number of K and finds the elbow point.
- **Silhouette method –** The silhouette method calculates similarity/dissimilarity score between their assigned cluster and the next best (i.e, nearest) cluster for each of the data points.

Moreover, there are also other techniques along with the above-mentioned ones to find the optimal no of k.

## **What is the training and testing complexity of the K means Algorithm?**

**Training complexity in terms of Big-O notation:**

If we use Lloyd’s algorithm, the complexity for training is: **“K\*I\*N\*M”**

where,

**K:** It represents the number of clusters

**I:** It represents the number of iterations

**N:** It represents the sample size

**M:** It represents the number of variables

**Conclusion:** There is a significant Impact on capping the number of iterations.

**Predicting complexity in terms of Big-O notation:**

**“K\*N\*M”**

Prediction needs to be computed for each record, the distance to each cluster and assigned to the nearest ones.

## ** Is it possible that the assignment of data points to clusters does not change between successive iterations in the K means Algorithm?**

When the K-Means algorithm has reached the local or global minima, it will not change the assignment of data points to clusters for two successive iterations during the algorithm run.

## ** Explain some cases where K means clustering fails to give good results.**

The K means clustering algorithm fails to give good results in the below-mentioned cases:

- When the dataset contains **outliers**
- When the **density spread** of data points across the data space is different.
- When the data points follow a **non-convex shape**.

![ k means clusters](https://lh6.googleusercontent.com/Ek0kFbKQ_HhZx8bNXT8tpsX7JZswpY7UAKl01tCwcttRhwQ8TQPFznQN1Yn5G6PHvHpfm2qFZRNhGiyVU7yhn-ei_QzfMjGTIkVIikbP3uyQyTrQmU3dYnNHjr0iRBUbQMp1WCIR)

## ** How to perform K means on larger datasets to make it faster?**

The idea behind this is **mini-batch k means**, which is an alternative to the traditional k means clustering algorithm that provides better performance for training on larger datasets.

It leverages the mini-batches of data, taken at random to update the cluster mean with a decreasing learning rate. For each data batch, the points are all first assigned to a cluster and then means are re-calculated. The cluster centres are then further re-calculated using **gradient descent**. This algorithm provides faster convergence than the typical k-means, but with a slightly different cluster output.

## ** What are the possible stopping conditions in the K means Algorithm?**

The following can be used as possible stopping conditions in K-Means clustering:

- **Max number of iterations has been reached**: This condition limits the runtime of the clustering algorithm, but in some cases, the quality of the clustering will be poor because of an insufficient number of iterations.
- **When RSS(within-cluster sum of squares) falls below a threshold**: This criterion ensures that the clustering is of the desired quality after termination. Practically in real-life problems, it’s a good practice to combine it with a bound on the number of iterations to guarantee convergence.
- **Convergence**: Points stay in the same cluster i.e., the algorithm has converged at the minima.
- **Stability**: Centroids of new clusters do not change.

## ** What is the effect of the number of variables on the K means Algorithm?**

The number of variables going into K means the algorithm has an impact on both the time(during training) and complexity(upon application) along with the behaviour of the algorithm as well.

This is also related to the **“Curse of dimensionality”**. As the dimensionality of the dataset increases, more and more examples become nearest neighbours of xt, until the choice of nearest neighbour is effectively random.

A key component of K means is that the distance-based computations are directly impacted by a large number of dimensions since the distances between a data point and its nearest and farthest neighbours can become equidistant in high dimension thereby resulting in reduced accuracy of distance-based analysis tools.

Therefore, we have to use the **Dimensionality reduction** techniques such as **Principal component analysis (PCA)**, or **Feature Selection Techniques**.

