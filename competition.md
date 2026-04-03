# Overview

Participants will receive an integer-indexed time series dataset (ts_index column), where each record is identified by a code, sub-code, sub-category and forecast horizon. Your objective is to train a model that generalizes robustly out-of-sample and accurately predicts future values for each combination (code, sub-code, sub-category, horizon). Your forecast will not use any data whose ts_index is greater than the ts_index of the forecast data. Submissions are ranked according to an aggregate out-of-sample metric calculated for all combinations.

## Description
You'll predict a continuous numerical value and be scored according to a measure inspired by the skill score.

The test set selected for scoring remains partially hidden (75%) throughout the process, to ensure a true out-of-sample evaluation.

# Evaluation

Submissions will be evaluated on the basis of this measure. The public ranking is calculated from approximately 25% of the test data. Final results will be based on the remaining 75%, so the final ranking may differ! Keep working, you could be the first even if you are not the first in the public rankings because of the risk of overfitting. The formula for calculating the metrics is attached:
with I a set of lines that will be 25% of the test for the public leaderboard and the remaining 75% for the private leaderboard. Below is a sample code that calculates the metric.
```python
def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))

```
This community contest does not follow the usual code-only format: you can submit your predictions as CSV files for evaluation. However, to ensure full reproducibility, we strongly recommend that you also provide a Kaggle notebook.

If your team finishes in the lead, you must submit executable code (with exact dependency versions, Python version) so that we can reproduce your results and check that no "data leakage" was exploited to game the ranking. To ensure fairness between participants, if your results are not easily reproducible you may be disqualified from monetary prize eligibility.

To avoid any data leakage, or otherwise look-forward, your code should predict ts_index t using only data from ts_index 0 to t, processing all data strictly sequentially.

## Submission

The submission must be a CSV file with a primary key (id) and a column (prediction). Predictions must be made on the test file (test.parquet). The submission file should look like this:

```csv
id,prediction
W2MW3G2L__STALY73S__9ZI8OAJB__1__2991 ; 5.764190326788755
83EG83KQ__R571RU17__PHHHVYZI__1__3353 ;   5.764190326788755
W2MW3G2L__STALY73S__Q101PRO5__3__2991 ;  5225125.5454
83EG83KQ__R571RU17__PZ9S1Z4V__1__3353 ;  4545.4545
```
Your notebook should be structured as follows.

    The imports
    The functions or classes used
    The code that will fit the model
    The prediction code

# Tips

Even though the exact times are hidden in the test data, it comes from a period after the training data. Feel free to focus on a specific window, for example by weighting the most recent periods, if this works best for your model.

Code, sub-code and sub-category have both similarities and differences. For instance similarities may exist between codes in same sub-category, or differences may relate between the same sub-codes across different codes. It is therefore essential to have an appropriate weighting between data in the same categories and data outside these categories.

One of the difficulties lies in the low signal-to-noise / ratio. Another difficulty lies in the fact that the underlying process is probably not completely stable with ts_index/time.

You may use all data that we provide or, in the case of external data, you may use it but it must comply with the rules set out in section 6. External data and tools.

Efforts in this competition can be directed towards two key aspects:

    Data mining and feature analysis.
    Advanced modeling techniques.

Prizes

Total prizes available: $10,000, divided between the top 5 teams

1st place - $3,500
2nd place - $2,500
3rd place - $2,000
4th place - $1,000
5th place - $1,000

The top five candidates may be offered a job interview at a hedge fund delivering around 30% annual returns at the end of the competition. We offer competitive terms and use modern technologies in a stimulating environment.

Do not forget that scored published on the leaderboard are not the ultimate score as competitor with better score could be using data leakage in the leaderboard.

Note: for the top teams eligible for a prize, submission of a Jupyter Notebook capable of successfully generating the prediction results is mandatory to qualify for the prize. If we cannot reproduce your result locally (missing dependency, missing python version, use of external data that does not comply with section 6 - External data and tools -, use of data with a ts index greater than the ts index of the target), you will be disqualified.
If your code or model training requires a specific execution environment or hardware configuration, this must be communicated clearly.

You are NOT permitted to use the test data set to aid the modeling process, as all test data - with the exception of the current test data point to be predicted - contains future information that would not be available at the time of prediction in a real-world environment.

As with the use of future data, any external data set containing future information will be considered a violation of the rule. So make sure that everything you use would have been available at the time of test data prediction!
FAQ

If you have any questions about the Kaggle competition, you can post them in the discussion forum with the title [Question for the organization] + title.
We will publish the answer here as soon as possible if the question is relevant.


### 
Dataset Description

The dataset is a time-series/tabular dataset with the following columns:

    id
    A unique key constructed by concatenating code, sub_code, sub_category, horizon, and ts_index with a double underscore (__). This ensures each row is distinctly identifiable.

    code
    A unique identifier for the entity.

    sub_code
    A categorical attribute grouping entities into sub-families or segments.

    sub_category A categorical label describing the broad category to which the entity belongs.

    ts_index
    Integer timestamp of the observation: indicates when the features were recorded.

    horizon
    A categorical forecast-horizon group. Typical codes are: 1. 1 = short-term 2. 3 = medium-term 3. 10 = long-term 4. 25 = extra long-term These codes do not represent the difference between ts_index.

    weight
    A numeric weight for each row, used to compute in the evaluation metric. DO NOT USE AS A FEATURE.These weights are used in the loss function formula (w).

    feature_a, feature_b, …, feature_ch
    A set of 86 anonymized features.

Data shape
Each row represents one forecast instance for a particular combination of (code, sub_code, sub_category, ts_index, horizon) along with its associated feature values. All features can be fed directly into typical regression models. 

## train file 
```parquet
 	id 	code 	sub_code 	sub_category 	horizon 	ts_index 	feature_a 	feature_b 	feature_c 	feature_d 	... 	feature_ca 	feature_cb 	feature_cc 	feature_cd 	feature_ce 	feature_cf 	feature_cg 	feature_ch 	y_target 	weight
0 	W2MW3G2L__J0G2B0KU__PZ9S1Z4V__25__89 	W2MW3G2L 	J0G2B0KU 	PZ9S1Z4V 	25 	89 	29 	16.364093 	7.464023 	5.966933 	... 	-0.001686 	-0.105328 	-0.005045 	NaN 	-0.133697 	2.849819 	0.112068 	1 	-0.551324 	4.098257e+01
1 	W2MW3G2L__J0G2B0KU__PZ9S1Z4V__1__89 	W2MW3G2L 	J0G2B0KU 	PZ9S1Z4V 	1 	89 	53 	2.858806 	5.050617 	15.906651 	... 	-0.001686 	-0.105328 	-0.005045 	NaN 	-0.133697 	2.849819 	0.112068 	1 	-0.315583 	1.500754e+02
2 	W2MW3G2L__J0G2B0KU__PZ9S1Z4V__3__89 	W2MW3G2L 	J0G2B0KU 	PZ9S1Z4V 	3 	89 	51 	9.585452 	1.076268 	9.004147 	... 	-0.001686 	-0.105328 	-0.005045 	NaN 	-0.133697 	2.849819 	0.112068 	1 	-0.362894 	1.159536e+02
3 	W2MW3G2L__J0G2B0KU__PZ9S1Z4V__10__89 	W2MW3G2L 	J0G2B0KU 	PZ9S1Z4V 	10 	89 	44 	8.840588 	15.034634 	4.170780 	... 	-0.001686 	-0.105328 	-0.005045 	NaN 	-0.133697 	2.849819 	0.112068 	1 	-0.667023 	6.457307e+01
4 	W2MW3G2L__J0G2B0KU__PZ9S1Z4V__25__90 	W2MW3G2L 	J0G2B0KU 	PZ9S1Z4V 	25 	90 	28 	2.303825 	7.696209 	12.896100 	... 	-0.001622 	-0.103809 	-0.005135 	NaN 	-0.174660 	2.738606 	0.109204 	1 	-0.437398 	4.194876e+01
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
5337409 	83EG83KQ__R571RU17__PHHHVYZI__3__3522 	83EG83KQ 	R571RU17 	PHHHVYZI 	3 	3522 	2 	2.756940 	2.611558 	14.494289 	... 	-0.000114 	-0.000196 	-0.009698 	-0.002835 	0.204607 	0.025985 	0.083269 	0 	0.000024 	2.292255e+08
5337410 	83EG83KQ__R571RU17__PHHHVYZI__3__3523 	83EG83KQ 	R571RU17 	PHHHVYZI 	3 	3523 	1 	3.560270 	2.992240 	2.049503 	... 	-0.000112 	-0.000199 	-0.009369 	-0.002780 	0.236836 	0.024961 	0.075778 	0 	-0.000901 	2.245107e+08
5337411 	83EG83KQ__R571RU17__PHHHVYZI__1__3523 	83EG83KQ 	R571RU17 	PHHHVYZI 	1 	3523 	1 	7.978713 	8.346381 	11.893113 	... 	-0.000112 	-0.000199 	-0.009369 	-0.002780 	0.236836 	0.024961 	0.075778 	0 	-0.000407 	3.544457e+08
5337412 	83EG83KQ__R571RU17__PHHHVYZI__1__3524 	83EG83KQ 	R571RU17 	PHHHVYZI 	1 	3524 	0 	3.120271 	13.186458 	0.747004 	... 	-0.000101 	-0.000187 	-0.008865 	-0.002522 	0.289178 	0.024614 	0.068916 	0 	-0.000124 	3.595513e+08
5337413 	83EG83KQ__R571RU17__PHHHVYZI__3__3524 	83EG83KQ 	R571RU17 	PHHHVYZI 	3 	3524 	0 	16.414658 	2.483409 	8.865080 	... 	-0.000101 	-0.000187 	-0.008865 	-0.002522 	0.289178 	0.024614 	0.068916 	0 	-0.000699 	2.231641e+08

5337414 rows × 94 columns
```

## Test file 
```parquet
id 	code 	sub_code 	sub_category 	horizon 	ts_index 	feature_a 	feature_b 	feature_c 	feature_d 	... 	feature_by 	feature_bz 	feature_ca 	feature_cb 	feature_cc 	feature_cd 	feature_ce 	feature_cf 	feature_cg 	feature_ch
0 	W2MW3G2L__495MGHFJ__PZ9S1Z4V__3__3647 	W2MW3G2L 	495MGHFJ 	PZ9S1Z4V 	3 	3647 	95 	10.365266 	3.209321 	8.109339 	... 	-0.000832 	-0.032241 	-0.000830 	-0.058961 	-0.002774 	-0.001480 	-0.256460 	1.665532 	0.071324 	2
1 	W2MW3G2L__495MGHFJ__PZ9S1Z4V__10__3647 	W2MW3G2L 	495MGHFJ 	PZ9S1Z4V 	10 	3647 	88 	2.571477 	15.234848 	16.505699 	... 	-0.000832 	-0.032241 	-0.000830 	-0.058961 	-0.002774 	-0.001480 	-0.256460 	1.665532 	0.071324 	2
2 	W2MW3G2L__495MGHFJ__PZ9S1Z4V__25__3647 	W2MW3G2L 	495MGHFJ 	PZ9S1Z4V 	25 	3647 	71 	5.524709 	6.931663 	8.939537 	... 	-0.000832 	-0.032241 	-0.000830 	-0.058961 	-0.002774 	-0.001480 	-0.256460 	1.665532 	0.071324 	2
3 	W2MW3G2L__495MGHFJ__PZ9S1Z4V__1__3647 	W2MW3G2L 	495MGHFJ 	PZ9S1Z4V 	1 	3647 	97 	10.293758 	14.893660 	9.435544 	... 	-0.000832 	-0.032241 	-0.000830 	-0.058961 	-0.002774 	-0.001480 	-0.256460 	1.665532 	0.071324 	2
4 	W2MW3G2L__495MGHFJ__PZ9S1Z4V__10__3648 	W2MW3G2L 	495MGHFJ 	PZ9S1Z4V 	10 	3648 	87 	14.776194 	7.701180 	6.228968 	... 	-0.000844 	-0.032988 	-0.000841 	-0.059835 	-0.002838 	-0.001501 	-0.242240 	1.671890 	0.071100 	2
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
1447102 	83EG83KQ__VYN97209__PHHHVYZI__3__4305 	83EG83KQ 	VYN97209 	PHHHVYZI 	3 	4305 	1 	16.314806 	16.075016 	4.102829 	... 	-0.003616 	-0.000562 	-0.000203 	-0.000326 	-0.017148 	-0.005026 	0.167489 	0.022252 	0.072114 	0
1447103 	83EG83KQ__VYN97209__PHHHVYZI__1__4306 	83EG83KQ 	VYN97209 	PHHHVYZI 	1 	4306 	2 	4.681250 	7.401711 	7.197559 	... 	-0.003743 	-0.000576 	-0.000231 	-0.000337 	-0.017592 	-0.005727 	0.171962 	0.021623 	0.069115 	0
1447104 	83EG83KQ__VYN97209__PHHHVYZI__3__4306 	83EG83KQ 	VYN97209 	PHHHVYZI 	3 	4306 	0 	5.372833 	13.592936 	16.349126 	... 	-0.003743 	-0.000576 	-0.000231 	-0.000337 	-0.017592 	-0.005727 	0.171962 	0.021623 	0.069115 	0
1447105 	83EG83KQ__VYN97209__PHHHVYZI__1__4307 	83EG83KQ 	VYN97209 	PHHHVYZI 	1 	4307 	1 	7.543404 	3.597098 	11.375947 	... 	-0.003897 	-0.000650 	-0.000262 	-0.000351 	-0.019842 	-0.006515 	0.196026 	0.021906 	0.067739 	0
1447106 	83EG83KQ__VYN97209__PHHHVYZI__1__4308 	83EG83KQ 	VYN97209 	PHHHVYZI 	1 	4308 	0 	2.557012 	9.656470 	0.752347 	... 	-0.005843 	-0.000819 	-0.000319 	-0.000526 	-0.025000 	-0.007918 	0.207929 	0.021718 	0.065738 	0

1447107 rows × 92 columns

```


## Host insides 

```
Q1:
User: 
About test data split 25% - 75% (public/private leaderboard)

For the leaderboard split (25% public, 75% private), is the test data divided randomly or in sequential order according to ts_index?

Based on my validation experiments, I believe the public/private split is sequential by ts_index, not random.

My validation on the last 10% of training data (ts_index 3241-3601) gives a score of 0.19, which closely matches my public LB score of 0.20.

However, when I validate on the last 20% or use time-series cross-validation (multiple 1200 ts_index validation sets), my validation score drops to around 0.11 (mean of fold scores)

Host: 
Thank you for your question about the leaderboard split. To clarify:
Sequential Split Details

The public 25% of the test data immediately follows the training data chronologically (by ts_index). The private 75% then immediately follows the public 25%, maintaining the sequential order.

In other words:

Training data → Public 25% → Private 75% This design ensures that the public leaderboard reflects performance on data that is temporally adjacent to the training set, while the private leaderboard tests generalization to the most recent (and unobserved) portion of the data. If you have further questions or need additional clarification, feel free to ask!

Best
```

---

```
Q2:
User:
what is sub-code and sub_category

Can you give more information about what code, sub-code, sub-category represents? It is very confusing what sub-code and sub_category represents because every code_subcode has 5 different sub-category. if it represents category of code entity why not there is only one unique value for one code? in the current data to get the unique timeseries we have to get unique code_subcode_subcategory combos which comes down to very few training rows per timeseries. since the propiretery features are already nonymized it would be helpfull if you can atleast give more detail about code, sub-code, sub_category. @adatacompany 

Host:
Hi Meet,

Thank you for your thoughtful question about the identifier columns (code, sub-code, sub-category). I understand how the structure of these fields can be confusing, especially when trying to interpret their relationships or map them to unique time series.
Key Points to Clarify:
Hierarchy and Relationships:

The code, sub-code, and sub-category fields are designed to capture hierarchical or relational information. A single sub-code may indeed appear across multiple codes, and a single sub-category may appear across multiple code_sub-code pairs. This reflects real-world relationships where entities may share attributes or belong to broader groups.
Why Multiple Sub-Categories per Code/Sub-Code?

The presence of multiple sub-categories for a given code_sub-code combination suggests that the data captures granular variations within those groups. For example, the sub-category could represent specific variants or context that is consistent across code-sub_code pairs.
Unique Time Series:

As you’ve observed, the unique time series are defined by the combination of code_sub-code_sub-category. While this may result in fewer training rows per series, it ensures that each series represents a distinct context or entity. This granularity is intentional to preserve the integrity of the relationships in the data.
Anonymization and Proprietary Features:

Since the proprietary features are anonymized, we cannot provide direct mappings or definitions for these identifiers. However, the structure is designed to allow participants to explore relationships (e.g., between codes sharing a sub-code or between sub-codes sharing a sub-category) as part of the modeling challenge.
Suggestions for Analysis:
Grouping Strategies:

Consider whether aggregating or grouping by certain levels (e.g., sub-code or sub-category) could help you leverage shared patterns across related time series.
Feature Engineering:

The relationships between these fields might be useful for creating derived features (e.g., counting unique sub-categories per sub-code or vice versa).

Note: I cannot guarantee that the suggestions above will necessarily lead to a better score, but they may be worth exploring as part of your analysis.

We appreciate your engagement and hope this helps clarify the design of these identifiers. If you have further questions or ideas for how we might improve the documentation, feel free to share!

Best
```

---

```
Q3:
User: 
About Target Variable, Embargo, and Horizon Values

Target Variable Definition

The competition description mentions we need to "predict a continuous numerical value," but I couldn't find specific information about what the target variable represents. Could you clarify:

    What does the target variable represent? (e.g., future price change, return, volatility, etc.)
    Is this information intentionally anonymized, or can you provide more context?

Cross-Validation Embargo/Gap

The rules clearly state: "Your forecast will not use any data whose ts_index is greater than the ts_index of the forecast data" "predict ts_index t using only data from ts_index 0 to t"

Given the presence of horizon values (1, 3, 10, 25), should we implement a gap/embargo in our time series cross-validation? For example:

    When using horizon=25, should we maintain at least a 25 ts_index gap between train and validation sets?
    Or is a simple time-based split sufficient without additional embargo?

Horizon Values Clarification

The description states: "horizon: A categorical forecast-horizon group […] These codes do not represent the difference between ts_index"

This suggests horizon=10 doesn't mean "10 ts_index steps ahead." However, it's unclear:

    Do these horizon values have any relationship to the actual temporal distance of predictions?
    Should horizon be treated purely as a categorical feature with no temporal meaning?

Understanding these aspects would help ensure our models comply with the competition rules and avoid data leakage. @adatacompany 

Host:

Hello Onur,

Target Variable Definition: Yes it is intentionnaly anonymized. The target is related to P&L over the horizon period.

Cross-Validation Embargo/Gap

For an ts_index=i, you can use all date from ts_index=1 up to ts_index=i. You do not need any additional embargo as targets for horizon (1,3,10,25) start their P&L at the end of ts_index=i

Horizon Values Clarification (1,3,5,25) are related to an horizon at (1,3,5,25). 

```

---


