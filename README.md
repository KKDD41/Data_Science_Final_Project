# Data Science Final Project.

## Task Overview.

### Assignment.
Build a model for binary (positive / negative) sentiment analysis of movies text reviews. Two datasets `train.csv` and `test.csv` were given for model's training and evaluation respectively.

### Code overview.

Initial data analysis and experiments were done in the notebooks (`./notebooks/` folder):
- `EDA.ipynb`
- `data_preprocessing.ipynb`
- `models_training_and_evaluation.ipynb`

## Exploratory Data Analysis.

### General dataset characteristics.

- 40'000 rows in training dataset in total. There are 272 duplicated rows and 266 duplicated reviews in total.
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>review</th>
      <td>40000</td>
      <td>39728</td>
      <td>Loved today's show!!! It was a variety and not...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>sentiment</th>
      <td>40000</td>
      <td>2</td>
      <td>positive</td>
      <td>20000</td>
    </tr>
  </tbody>
</table>
</div>

- There are no `null` or empty string values in both `review` and `sentiment` columns.

- There are no same reviews with different sentimental, therefore no logical inconsistency persists in train dataset.  

- There are equal amount of positive and negative sentiments (20'000 rows for both).
![png](outputs/figures/sentiment_count_histogram.png)

### Feature Engineering and Text Analysis.

The following numerical features of reviews were investigated: 
```
'number_of_words'
'number_of_chars'
'percentage_of_signs'
'number_of_excl_marks'
'number_of_question_marks'
'number_of_ellipses'
'number_of_uppercase_words'
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>number_of_words</th>
      <td>40000.0</td>
      <td>231.362750</td>
      <td>171.083908</td>
      <td>4.000000</td>
      <td>126.000000</td>
      <td>173.00000</td>
      <td>282.000000</td>
      <td>2470.000000</td>
    </tr>
    <tr>
      <th>number_of_chars</th>
      <td>40000.0</td>
      <td>1310.549450</td>
      <td>987.955229</td>
      <td>41.000000</td>
      <td>699.000000</td>
      <td>971.00000</td>
      <td>1595.000000</td>
      <td>13704.000000</td>
    </tr>
    <tr>
      <th>percentage_of_signs</th>
      <td>40000.0</td>
      <td>21.977625</td>
      <td>1.825969</td>
      <td>11.764706</td>
      <td>20.805369</td>
      <td>21.83136</td>
      <td>22.940277</td>
      <td>87.311178</td>
    </tr>
    <tr>
      <th>number_of_excl_marks</th>
      <td>40000.0</td>
      <td>0.971950</td>
      <td>2.957310</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>282.000000</td>
    </tr>
    <tr>
      <th>number_of_question_marks</th>
      <td>40000.0</td>
      <td>0.645175</td>
      <td>1.495052</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>number_of_ellipses</th>
      <td>40000.0</td>
      <td>0.499400</td>
      <td>1.580463</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>number_of_uppercase_words</th>
      <td>40000.0</td>
      <td>4.878900</td>
      <td>5.585357</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.00000</td>
      <td>6.000000</td>
      <td>151.000000</td>
    </tr>
  </tbody>
</table>
</div>

- Length of characters / words in review are skewed left. Mean number of words in review is 231 and mean number of chars -- 1310. While maximum number of chars is more than 13'000. Maximum number of words is 2'500.

![png](outputs/figures/number_of_chars_distribution.png) ![png](outputs/figures/number_of_words_distribution.png)

- There is no high correlation between them (considering Pearson, Kendall and Spearman correlation coefficients), except obvious dependency between number of words and characters.

<img src="./outputs/figures/Pearson_cm.png" width="30%"/> <img src="./outputs/figures/Kendall_cm.png" width="30%"/> <img src="./outputs/figures/Spearman_cm.png" width="30%"/> 

- Average ratio of non-alphabetical chars in review is 21% (which is pretty high). Since in texts appears fragments `<br />`, `blablablabla+`, ` >>>>>>> `, `*[word]*`, `........`, `?[word]?`, `[word]-[word]-[word]` and other noise, therefore number of particular characters / marks were kept as numerical features, and all other non-alphabetical signs were removed during text preprocessing.
![png](outputs/figures/Percentage_of_signs_in_review.png)

- Were obtained, that `number_of_ellipses` and `number_of_question_marks` are higher for negative sentiments. For other `numerical_review_features` box-plots showed no significant difference in distributions between positive and negative sentiments.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>negative</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">number_of_words</th>
      <th>count</th>
      <td>20000.000000</td>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>230.186850</td>
      <td>232.538650</td>
    </tr>
    <tr>
      <th>std</th>
      <td>165.642483</td>
      <td>176.353828</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>128.000000</td>
      <td>125.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>174.000000</td>
      <td>172.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>280.000000</td>
      <td>284.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1522.000000</td>
      <td>2470.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">number_of_chars</th>
      <th>count</th>
      <td>20000.000000</td>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1298.143300</td>
      <td>1322.955600</td>
    </tr>
    <tr>
      <th>std</th>
      <td>950.224379</td>
      <td>1024.170719</td>
    </tr>
    <tr>
      <th>min</th>
      <td>41.000000</td>
      <td>65.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>705.000000</td>
      <td>692.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>974.000000</td>
      <td>968.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1576.000000</td>
      <td>1614.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8969.000000</td>
      <td>13704.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">percentage_of_signs</th>
      <th>count</th>
      <td>20000.000000</td>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22.163721</td>
      <td>21.791530</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.776076</td>
      <td>1.856013</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.764706</td>
      <td>14.925373</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.985011</td>
      <td>20.629241</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.025873</td>
      <td>21.639938</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>23.125000</td>
      <td>22.758413</td>
    </tr>
    <tr>
      <th>max</th>
      <td>38.847858</td>
      <td>87.311178</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">number_of_excl_marks</th>
      <th>count</th>
      <td>20000.000000</td>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.009400</td>
      <td>0.934500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.540263</td>
      <td>3.322057</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>70.000000</td>
      <td>282.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">number_of_question_marks</th>
      <th>count</th>
      <td>20000.000000</td>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.905000</td>
      <td>0.385350</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.825881</td>
      <td>1.000802</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>35.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">number_of_ellipses</th>
      <th>count</th>
      <td>20000.000000</td>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.599500</td>
      <td>0.399300</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.664343</td>
      <td>1.485183</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>48.000000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">number_of_uppercase_words</th>
      <th>count</th>
      <td>20000.000000</td>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.171500</td>
      <td>4.586300</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.608267</td>
      <td>5.547079</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>151.000000</td>
      <td>126.000000</td>
    </tr>
  </tbody>
</table>
</div> 

![png](outputs/figures/Boxplots_for_numerical_features.png)


## Text Preprocessing.



