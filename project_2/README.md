# Problem Statement: 

Our client, Iowa Appraisals and Research Inc, faces a loss of customer confidence due to an unreliable property appraisal. As such, we attempt to build a predictive model to more accurately appraise the value of a property.


# Background: 

In the process of purchasing or selling a property, it must first be appraised for both parties to have an objective understanding of its value. In such cases, objectivity can be supported with data of the features of each property, where some features such as lot square footage are quantitative while others such as property quality are qualitative. Combining the tools in data science, we can provide a clearer understanding for what makes a property more valuable, and create a model to appropriately appraise the property value.


# Datasets Used:

|No.|File Name|Description|
|-|-|-|
|1.| train.csv| Ames Housing Training Data (provided by General Assembly)|
|2.| test.csv| Ames Housing Test Data ex Target Column (provided by General Assembly)|


# Data Dictionary:

|Column	|		Type		|Description|
|-|-|-|
|Lot_Area		|int64	|	Lot size (sqft)|
|MS_SubClass	|	object	|	Property Zoning Classification|
|Overall_Qual	|int64	|	Overall Quality of Property|
|Exter_Qual	|	object	|	Property External Quality|
|Total_Bsmt_SF	|float64	|	Total Basement Area (sqft)|
|Heating		|	object	|	Heating Type|
|Neighborhood|	object	|	Property Neighborhood Name|
|Gr_Liv_Area|		int64	|	Size of Living Area (sqft)|
|Garage_Area	|	float64	|	Size of Garage (sqft)|
|SalePrice	|	int64	|	Sale Price of Property|
|SP_LA		|	float	|	Sale Price per Lot Area ($/sqft)|
|House_Sold_Age|	int64		|Age of House at Sale (years)|
|GLA_OQ		|	int64	|	Grade Living Area * Overall Property Quality|

# Methodology

## Part 1: Data Cleaning

We conduct several data cleaning for readability and processing purposes. When we first read the dataset, we found many missing values in multiple columns. As and when we can, we impute the values as zero and provide justification for such cases. In general, a missing value denotes the absence of such features. For instance, the absence of a garage had originally been inputted as a 'nan' value. In such cases, we replace the 'nan' values with '0' to denote the absence of a garage and enable the datapoint to be processed in our model.

Where we are unable to impute zeroes, we conducted a linear regression imputation. A total of one variable was imputed this way: Lot Frontage. We find that the variables such as lot area used to impute this variable provide meaningful imputation to the lot frontage.

In addition, there were several qualitative variables which have an ordinal nature. For instance, under Exterior Condition, the ordinal values Ex, Gd, TA, Fa, Po denoted the levels of favorability of such feature. In such cases, we modified the values into 5, 4, 3, 2, 1 respectively, where the higher the numeric value, the greater the favorability of the feature.

We created several dummies based on the presence or absence of a feature. For instance, under the variable 'Alley', we find two unique values, 'Gravel' and 'Paved'. We dummified this variable into '1's and '0's, where '1' represents the presence of a Gravel or Paved Alley separately, and '0' on both columns denotee the absence of an alley.


## Part 2: Exploratory Data Analysis

We conducted several EDAs to visualize the effect of some variables and drop outliers.

Visualization provides a meaningful insight onto the baseline effects of some variables. For instance, by visualizing Zoning class on boxplots, we find that there is an effect in Zoning to the median and interquartile sale price of a property per square footage.

We also dropped several outliers in which were large deviations and unlikely to be meaningful to our model. In such cases, the negative predictive value of such observations exceed that of its positive value of inclusion. For instance, we found some datapoints with outlier lot area values as compared to its expected sale price. Such outliers may be due to external conditions which are not quantified in our original dataset. As such, we have dropped such observations.


## Part 3: Predictive Modeling

We create several models based on differing purposes.

Our first model takes into account just one variable, lot area, and predicts sale price based on this variable. Lot area was selected as our single variable due to its inherent 'go-to' nature in estimating property values. This provides us a baseline model where other models are compared to.

Our second model takes into account only numerical variables in an elastic net model. The purpose of this model is to find which regularization model would be more appropriate: Ridge or Lasso. Through testing a wide range of alpha values, we found that a Lasso model is more appropriate to use and will provide a more meaningful predictive model.

Our third model is a lasso model based on only numerical variables. The purpose of this model is as a baseline lasso model to compare other lasso models and improve upon them.

Our fourth model is a lasso model based on a mix of quantitative and qualitative variables. We find that the model has improved with the inclusion of several qualitative features, and a selective exclusion of quantitative variables. We tested this model on the Kaggle target and found that while the model had performed relatively well in both our training and testing set, it had a larger RMSE in our Kaggle targets.

Our fifth model is a lasso model based on a subset of features used in our fourth model. We find that although the model had worsened as compared to our fourth model, the tradeoff may have been favorable overall as we had used a significantly smaller number of variables (7 compared to 11). Testing this model on our Kaggle target, we see an improvement in RMSE scores. This suggests that we had significantly improved our model variance by excluding the 4 variables which had contributed to noise rather than its expected predictive value.

Our sixth and final model is a lasso model based on all features used in our fifth model, with an addition of our engineered feature which combines grade living area and overall property quality. We find that this is our best model in all metrics: R^2 and RMSE on all training, testing and Kaggle scores. We attribute this drastic improvement to the inclusion of our featured engineer, which provides a significant amount of predictive value.

## Part 4: Predictive Model Target Application

In this section, we cleaned and processed the data in the same steps as we had in Part 1, and ran our fourth, fifth and sixth predictive model from Part 3 to the Kaggle target dataset. We find that a limited number of observations (8) had provided a very meaningful amount of predictive value to our target data due to it's value as high-impact variables.


## Part 5: Conclusion and Recommendation

We conclude by observing a few things. Firstly, our model has greater predictive value than that of the baseline model. Secondly, a lasso model is more appropriate than a ridge model, as it completely excludes several variables instead of minimizing its value. Thirdly, we find that a combination of quantitative and qualitative variables provide a more meaningful predictive value than using only quantitative values. Next, we find that the inclusion of a limited number of high-impact variables provide greater value than a larger number of lower-impact variables due to the noise caused by the extra variables without providing meaningful predictive power.

Finally, we recommend the usage of our predictive model in appraising property values. We also note that several qualitative factors such as Neighborhood had provided a significant amount of predictive value, and it may be prudent to collect other data tangential to Neighborhood, such as HOA fees, school districts, proximity to various destinations such as highways, and public safety-related data such as crime rates.
