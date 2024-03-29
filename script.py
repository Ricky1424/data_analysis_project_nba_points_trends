# Importing necessary libraries
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Setting options for better readability
import codecademylib3
np.set_printoptions(suppress=True, precision=2)

# Reading the NBA data
nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

# Displaying first few rows of each subset
print(nba_2010.head())
print(nba_2014.head())

# Step 1: Extracting points scored by Knicks and Nets in 2010
knicks_pts_10 = nba_2010.pts[nba_2010.fran_id == 'Knicks']
nets_pts_10 = nba_2010.pts[nba_2010.fran_id == 'Nets']

# Step 2: Calculating mean difference between Knicks and Nets points in 2010
diff_means_2010 = knicks_pts_10.mean() - nets_pts_10.mean()
print(f'Mean difference 2010: {diff_means_2010}')

# Step 3: Plotting overlapping histograms for points scored by Knicks and Nets in 2010
plt.hist(knicks_pts_10, color='red', label='Knicks', normed=False, alpha=0.5)
plt.hist(nets_pts_10, color='blue', label='Nets', normed=False, alpha=0.5)
plt.legend()
plt.show()
plt.close()

# Step 4: Similar analysis for 2014 data
knicks_pts_14 = nba_2014.pts[nba_2014.fran_id == 'Knicks']
nets_pts_14 = nba_2014.pts[nba_2014.fran_id == 'Nets']
diff_means_2014 = knicks_pts_14.mean() - nets_pts_14.mean()
print(f'Mean difference 2014: {diff_means_2014}')
plt.hist(knicks_pts_14, color='red', label='Knicks', normed=False, alpha=0.5)
plt.hist(nets_pts_14, color='blue', label='Nets', normed=False, alpha=0.5)
plt.legend()
plt.show()
plt.close()

# Step 5: Investigating relationship between franchise and points scored per game in 2010
sns.boxplot(data=nba_2010, x='fran_id', y='pts')
plt.show()
plt.close()

# Step 6: Calculating contingency table of frequencies for game results and locations
location_result_freq = pd.crosstab(nba_2010.game_result, nba_2010.game_location)
print(location_result_freq)

# Step 7: Converting frequency table to proportions
location_results_prop = location_result_freq / len(nba_2010)
print(location_results_prop)

# Step 8: Calculating expected contingency table and Chi-Square statistic
chi2, pval, dof, expected = chi2_contingency(location_result_freq)
print(np.round(expected))

# Step 9: Calculating covariance between forecast and point difference in 2010
point_diff_forecast_cov = np.cov(nba_2010.forecast, nba_2010.point_diff)
print(point_diff_forecast_cov)

# Step 10: Calculating correlation between forecast and point difference in 2010
point_diff_forecast_corr, p = pearsonr(nba_2010.forecast, nba_2010.point_diff)
print(point_diff_forecast_corr)

# Step 11: Generating scatter plot of forecast and point difference
plt.scatter(x=nba_2010.forecast, y=nba_2010.point_diff)
plt.xlabel('Forecast')
plt.ylabel('Point Diff')
plt.show()
