# day_data_overview

### dataset_info


=== Dataset Information ===

| Info                      | Details               |
|---------------------------|-----------------------|
| Total Shape               | 731 rows, 16 columns  |
| Range Index               | 0 to 730, 731 entries |
| Columns                   | 16 columns            |
| Memory Usage              | 130981 bytes          |
| Total Duplicated Rows     | 0 duplicates          |
| Missing Values Count      | 0 missing values      |
| Missing Values Percentage | 0.00% missing values  |

Data types with counts of columns:

| Data Type   |   Count of Columns |
|-------------|--------------------|
| int64       |                 11 |
| float64     |                  4 |
| object      |                  1 |

Numerical and Categorical Variable Counts:

| Variable Type         |   Count |
|-----------------------|---------|
| Numerical Variables   |      15 |
| Categorical Variables |       1 |

No duplicated rows found.


### data_overview


=== Data Overview Table ===


Overview for Data Type: int64

|    | Column Name   | Data Type   |   Unique Count |   Missing Count |   Missing Percentage | Missing Value Category   |
|----|---------------|-------------|----------------|-----------------|----------------------|--------------------------|
|  0 | instant       | int64       |            731 |               0 |                    0 | No Missing Values        |
|  1 | season        | int64       |              4 |               0 |                    0 | No Missing Values        |
|  2 | yr            | int64       |              2 |               0 |                    0 | No Missing Values        |
|  3 | mnth          | int64       |             12 |               0 |                    0 | No Missing Values        |
|  4 | holiday       | int64       |              2 |               0 |                    0 | No Missing Values        |
|  5 | weekday       | int64       |              7 |               0 |                    0 | No Missing Values        |
|  6 | workingday    | int64       |              2 |               0 |                    0 | No Missing Values        |
|  7 | weathersit    | int64       |              3 |               0 |                    0 | No Missing Values        |
|  8 | casual        | int64       |            606 |               0 |                    0 | No Missing Values        |
|  9 | registered    | int64       |            679 |               0 |                    0 | No Missing Values        |
| 10 | cnt           | int64       |            696 |               0 |                    0 | No Missing Values        |

Overview for Data Type: float64

|    | Column Name   | Data Type   |   Unique Count |   Missing Count |   Missing Percentage | Missing Value Category   |
|----|---------------|-------------|----------------|-----------------|----------------------|--------------------------|
|  0 | temp          | float64     |            499 |               0 |                    0 | No Missing Values        |
|  1 | atemp         | float64     |            690 |               0 |                    0 | No Missing Values        |
|  2 | hum           | float64     |            595 |               0 |                    0 | No Missing Values        |
|  3 | windspeed     | float64     |            650 |               0 |                    0 | No Missing Values        |

Overview for Data Type: object

|    | Column Name   | Data Type   |   Unique Count |   Missing Count |   Missing Percentage | Missing Value Category   |
|----|---------------|-------------|----------------|-----------------|----------------------|--------------------------|
|  0 | dteday        | object      |            731 |               0 |                    0 | No Missing Values        |

### categorize_columns


=== Binary Columns ===

| Binary Column List   | Distinct Value Count (Binary)   |
|----------------------|---------------------------------|

=== Multi-Category Columns ===

|    | Multi Categories   |   Distinct Value Count (Multi) |
|----|--------------------|--------------------------------|
|  0 | dteday             |                            731 |

### outliers_summary


=== Outliers Summary ===

|   Index | Column     |   Outlier Count | Percentage   |
|---------|------------|-----------------|--------------|
|       0 | instant    |               0 | 0.00%        |
|       1 | season     |               0 | 0.00%        |
|       2 | yr         |               0 | 0.00%        |
|       3 | mnth       |               0 | 0.00%        |
|       4 | holiday    |              21 | 2.87%        |
|       5 | weekday    |               0 | 0.00%        |
|       6 | workingday |               0 | 0.00%        |
|       7 | weathersit |               0 | 0.00%        |
|       8 | temp       |               0 | 0.00%        |
|       9 | atemp      |               0 | 0.00%        |
|      10 | hum        |               2 | 0.27%        |
|      11 | windspeed  |              13 | 1.78%        |
|      12 | casual     |              44 | 6.02%        |
|      13 | registered |               0 | 0.00%        |
|      14 | cnt        |               0 | 0.00%        |

### summary_statistics_all

Summary Statistics for All Numeric Columns:

| Statistic                   |         instant |        season |            yr |          mnth |       holiday |       weekday |    workingday |    weathersit |          temp |         atemp |          hum |     windspeed |           casual |     registered |            cnt |
|-----------------------------|-----------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|--------------|---------------|------------------|----------------|----------------|
| Count                       |   731           | 731           | 731           | 731           | 731           | 731           | 731           | 731           | 731           | 731           | 731          | 731           |    731           |  731           |  731           |
| Unique                      |   731           |   4           |   2           |  12           |   2           |   7           |   2           |   3           | 499           | 690           | 595          | 650           |    606           |  679           |  696           |
| Mean                        |   366           |   2.49658     |   0.500684    |   6.51984     |   0.0287278   |   2.99726     |   0.683995    |   1.39535     |   0.495385    |   0.474354    |   0.627894   |   0.190486    |    848.176       | 3656.17        | 4504.35        |
| Std                         |   211.166       |   1.11081     |   0.500342    |   3.45191     |   0.167155    |   2.00479     |   0.465233    |   0.544894    |   0.183051    |   0.162961    |   0.142429   |   0.0774979   |    686.622       | 1560.26        | 1937.21        |
| Min                         |     1           |   1           |   0           |   1           |   0           |   0           |   0           |   1           |   0.0591304   |   0.0790696   |   0          |   0.0223917   |      2           |   20           |   22           |
| 25%                         |   183.5         |   2           |   0           |   4           |   0           |   1           |   0           |   1           |   0.337083    |   0.337842    |   0.52       |   0.13495     |    315.5         | 2497           | 3152           |
| 50%                         |   366           |   3           |   1           |   7           |   0           |   3           |   1           |   1           |   0.498333    |   0.486733    |   0.626667   |   0.180975    |    713           | 3662           | 4548           |
| 75%                         |   548.5         |   3           |   1           |  10           |   0           |   5           |   1           |   2           |   0.655417    |   0.608602    |   0.730209   |   0.233214    |   1096           | 4776.5         | 5956           |
| Max                         |   731           |   4           |   1           |  12           |   1           |   6           |   1           |   3           |   0.861667    |   0.840896    |   0.9725     |   0.507463    |   3410           | 6946           | 8714           |
| Mode                        |     1           |   3           |   1           |   1           |   0           |   0           |   1           |   1           |   0.265833    |   0.654688    |   0.613333   |   0.10635     |    120           | 1707           | 1096           |
| Range                       |   730           |   3           |   1           |  11           |   1           |   6           |   1           |   2           |   0.802537    |   0.761826    |   0.9725     |   0.485071    |   3408           | 6926           | 8692           |
| IQR                         |   365           |   1           |   1           |   6           |   0           |   4           |   1           |   1           |   0.318333    |   0.27076     |   0.210209   |   0.0982645   |    780.5         | 2279.5         | 2804           |
| Variance                    | 44591           |   1.23389     |   0.250342    |  11.9157      |   0.0279407   |   4.01917     |   0.216442    |   0.29691     |   0.0335077   |   0.0265563   |   0.020286   |   0.00600592  | 471450           |    2.4344e+06  |    3.75279e+06 |
| Skewness                    |     0           |  -0.000384278 |  -0.00274161  |  -0.00814865  |   5.65422     |   0.0027416   |  -0.793147    |   0.957385    |  -0.054521    |  -0.131088    |  -0.0697834  |   0.677345    |      1.26645     |    0.0436588   |   -0.0473528   |
| Kurtosis                    |    -1.2         |  -1.3426      |  -2.00549     |  -1.20911     |  30.0525      |  -1.25428     |  -1.37469     |  -0.136467    |  -1.11886     |  -0.985131    |  -0.0645301  |   0.410922    |      1.32207     |   -0.713097    |   -0.811922    |
| Shapiro-Wilk Test Statistic |     0.954773    |   0.858217    |   0.636572    |   0.941272    |   0.154387    |   0.91728     |   0.585348    |   0.659033    |   0.965912    |   0.973839    |   0.993345   |   0.971232    |      0.885015    |    0.9843      |    0.980124    |
| Shapiro-Wilk Test p-value   |     3.35463e-14 |   4.06491e-25 |   1.24567e-36 |   2.11222e-16 |   6.63295e-49 |   1.49342e-19 |   1.99624e-38 |   8.86823e-36 |   5.14588e-12 |   3.74366e-10 |   0.00248085 |   8.42614e-11 |      7.26595e-23 |    4.61533e-07 |    2.08129e-08 |

### categorical_summary

Categorical Summary:

| Statistic      | dteday     |
|----------------|------------|
| Count          | 731        |
| Unique         | 731        |
| Top            | 2011-01-01 |
| Frequency      | 1          |
| Top Percentage | 0.14%      |



# hour_data_overview

### dataset_info


=== Dataset Information ===

| Info                      | Details                   |
|---------------------------|---------------------------|
| Total Shape               | 17379 rows, 17 columns    |
| Range Index               | 0 to 17378, 17379 entries |
| Columns                   | 17 columns                |
| Memory Usage              | 3250005 bytes             |
| Total Duplicated Rows     | 0 duplicates              |
| Missing Values Count      | 0 missing values          |
| Missing Values Percentage | 0.00% missing values      |

Data types with counts of columns:

| Data Type   |   Count of Columns |
|-------------|--------------------|
| int64       |                 12 |
| float64     |                  4 |
| object      |                  1 |

Numerical and Categorical Variable Counts:

| Variable Type         |   Count |
|-----------------------|---------|
| Numerical Variables   |      16 |
| Categorical Variables |       1 |

No duplicated rows found.


### data_overview


=== Data Overview Table ===


Overview for Data Type: int64

|    | Column Name   | Data Type   |   Unique Count |   Missing Count |   Missing Percentage | Missing Value Category   |
|----|---------------|-------------|----------------|-----------------|----------------------|--------------------------|
|  0 | instant       | int64       |          17379 |               0 |                    0 | No Missing Values        |
|  1 | season        | int64       |              4 |               0 |                    0 | No Missing Values        |
|  2 | yr            | int64       |              2 |               0 |                    0 | No Missing Values        |
|  3 | mnth          | int64       |             12 |               0 |                    0 | No Missing Values        |
|  4 | hr            | int64       |             24 |               0 |                    0 | No Missing Values        |
|  5 | holiday       | int64       |              2 |               0 |                    0 | No Missing Values        |
|  6 | weekday       | int64       |              7 |               0 |                    0 | No Missing Values        |
|  7 | workingday    | int64       |              2 |               0 |                    0 | No Missing Values        |
|  8 | weathersit    | int64       |              4 |               0 |                    0 | No Missing Values        |
|  9 | casual        | int64       |            322 |               0 |                    0 | No Missing Values        |
| 10 | registered    | int64       |            776 |               0 |                    0 | No Missing Values        |
| 11 | cnt           | int64       |            869 |               0 |                    0 | No Missing Values        |

Overview for Data Type: float64

|    | Column Name   | Data Type   |   Unique Count |   Missing Count |   Missing Percentage | Missing Value Category   |
|----|---------------|-------------|----------------|-----------------|----------------------|--------------------------|
|  0 | temp          | float64     |             50 |               0 |                    0 | No Missing Values        |
|  1 | atemp         | float64     |             65 |               0 |                    0 | No Missing Values        |
|  2 | hum           | float64     |             89 |               0 |                    0 | No Missing Values        |
|  3 | windspeed     | float64     |             30 |               0 |                    0 | No Missing Values        |

Overview for Data Type: object

|    | Column Name   | Data Type   |   Unique Count |   Missing Count |   Missing Percentage | Missing Value Category   |
|----|---------------|-------------|----------------|-----------------|----------------------|--------------------------|
|  0 | dteday        | object      |            731 |               0 |                    0 | No Missing Values        |

### categorize_columns


=== Binary Columns ===

| Binary Column List   | Distinct Value Count (Binary)   |
|----------------------|---------------------------------|

=== Multi-Category Columns ===

|    | Multi Categories   |   Distinct Value Count (Multi) |
|----|--------------------|--------------------------------|
|  0 | dteday             |                            731 |

### outliers_summary


=== Outliers Summary ===

|   Index | Column     |   Outlier Count | Percentage   |
|---------|------------|-----------------|--------------|
|       0 | instant    |               0 | 0.00%        |
|       1 | season     |               0 | 0.00%        |
|       2 | yr         |               0 | 0.00%        |
|       3 | mnth       |               0 | 0.00%        |
|       4 | hr         |               0 | 0.00%        |
|       5 | holiday    |             500 | 2.88%        |
|       6 | weekday    |               0 | 0.00%        |
|       7 | workingday |               0 | 0.00%        |
|       8 | weathersit |               3 | 0.02%        |
|       9 | temp       |               0 | 0.00%        |
|      10 | atemp      |               0 | 0.00%        |
|      11 | hum        |              22 | 0.13%        |
|      12 | windspeed  |             342 | 1.97%        |
|      13 | casual     |            1192 | 6.86%        |
|      14 | registered |             680 | 3.91%        |
|      15 | cnt        |             505 | 2.91%        |

### summary_statistics_all

Summary Statistics for All Numeric Columns:

| Statistic                   |         instant |          season |              yr |            mnth |              hr |          holiday |        weekday |       workingday |      weathersit |            temp |           atemp |             hum |       windspeed |          casual |     registered |             cnt |
|-----------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|------------------|----------------|------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|----------------|-----------------|
| Count                       | 17379           | 17379           | 17379           | 17379           | 17379           | 17379            | 17379          | 17379            | 17379           | 17379           | 17379           | 17379           | 17379           | 17379           | 17379          | 17379           |
| Unique                      | 17379           |     4           |     2           |    12           |    24           |     2            |     7          |     2            |     4           |    50           |    65           |    89           |    30           |   322           |   776          |   869           |
| Mean                        |  8690           |     2.50164     |     0.502561    |     6.53778     |    11.5468      |     0.0287704    |     3.00368    |     0.682721     |     1.42528     |     0.496987    |     0.475775    |     0.627229    |     0.190098    |    35.6762      |   153.787      |   189.463       |
| Std                         |  5017.03        |     1.10692     |     0.500008    |     3.43878     |     6.91441     |     0.167165     |     2.00577    |     0.465431     |     0.639357    |     0.192556    |     0.17185     |     0.19293     |     0.12234     |    49.305       |   151.357      |   181.388       |
| Min                         |     1           |     1           |     0           |     1           |     0           |     0            |     0          |     0            |     1           |     0.02        |     0           |     0           |     0           |     0           |     0          |     1           |
| 25%                         |  4345.5         |     2           |     0           |     4           |     6           |     0            |     1          |     0            |     1           |     0.34        |     0.3333      |     0.48        |     0.1045      |     4           |    34          |    40           |
| 50%                         |  8690           |     3           |     1           |     7           |    12           |     0            |     3          |     1            |     1           |     0.5         |     0.4848      |     0.63        |     0.194       |    17           |   115          |   142           |
| 75%                         | 13034.5         |     3           |     1           |    10           |    18           |     0            |     5          |     1            |     2           |     0.66        |     0.6212      |     0.78        |     0.2537      |    48           |   220          |   281           |
| Max                         | 17379           |     4           |     1           |    12           |    23           |     1            |     6          |     1            |     4           |     1           |     1           |     1           |     0.8507      |   367           |   886          |   977           |
| Mode                        |     1           |     3           |     1           |     5           |    16           |     0            |     6          |     1            |     1           |     0.62        |     0.6212      |     0.88        |     0           |     0           |     4          |     5           |
| Range                       | 17378           |     3           |     1           |    11           |    23           |     1            |     6          |     1            |     3           |     0.98        |     1           |     1           |     0.8507      |   367           |   886          |   976           |
| IQR                         |  8689           |     1           |     1           |     6           |    12           |     0            |     4          |     1            |     1           |     0.32        |     0.2879      |     0.3         |     0.1492      |    44           |   186          |   241           |
| Variance                    |     2.51706e+07 |     1.22527     |     0.250008    |    11.8252      |    47.809       |     0.0279442    |     4.02312    |     0.216626     |     0.408777    |     0.0370779   |     0.0295325   |     0.0372219   |     0.0149671   |  2430.99        | 22909          | 32901.5         |
| Skewness                    |     0           |    -0.0054157   |    -0.0102433   |    -0.00925325  |    -0.0106799   |     5.63854      |    -0.00299822 |    -0.785258     |     1.22805     |    -0.00602088  |    -0.0904289   |    -0.111287    |     0.574905    |     2.49924     |     1.5579     |     1.27741     |
| Kurtosis                    |    -1.2         |    -1.33425     |    -2.00013     |    -1.20188     |    -1.19802     |    29.7965       |    -1.256      |    -1.38353      |     0.350151    |    -0.941844    |    -0.845412    |    -0.826117    |     0.59082     |     7.571       |     2.75002    |     1.4172      |
| Shapiro-Wilk Test Statistic |     0.954925    |     0.859197    |     0.636615    |     0.942238    |     0.951477    |     0.154545     |     0.917127   |     0.586161     |     0.660587    |     0.978475    |     0.98137     |     0.98001     |     0.95891     |     0.706704    |     0.851127   |     0.873459    |
| Shapiro-Wilk Test p-value   |     1.04811e-57 |     2.65331e-81 |     4.2195e-104 |     1.71817e-62 |     4.15604e-59 |     8.00109e-127 |     8.3049e-70 |     1.88382e-107 |     2.3023e-102 |     1.31557e-44 |     2.98929e-42 |     2.15802e-43 |     5.70403e-56 |     1.06902e-98 |     1.4557e-82 |     6.48834e-79 |

### categorical_summary

Categorical Summary:

| Statistic      | dteday     |
|----------------|------------|
| Count          | 17379      |
| Unique         | 731        |
| Top            | 2011-01-01 |
| Frequency      | 24         |
| Top Percentage | 0.14%      |



