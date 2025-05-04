import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from scipy.stats import ks_2samp, norm, probplot, mannwhitneyu, ttest_rel, ttest_ind
import numpy as np
import pingouin as pg
import scipy.stats as stats


def bootstrap_ks_test_Z(data, feature, subsample_size1=100,subsample_size2 = 100, n_iterations=1000):

    group_0 = data[data['label'] == 0][feature].to_numpy()
    group_1 = data[data['label'] == 1][feature].to_numpy()

    p_values = []
    ks_statistics = []


    for _ in range(n_iterations):
        sample_0 = np.random.choice(group_0, size=min(subsample_size1, len(group_0)), replace=True)
        sample_1 = np.random.choice(group_1, size=min(subsample_size2, len(group_1)), replace=True)

        ks_stat, p_value = ks_2samp(sample_0, sample_1)
        p_values.append(p_value)
        ks_statistics.append(ks_stat)

    p_adjusted = multipletests(p_values, method='fdr_bh')[1]
    z_values = [norm.ppf(1 - p / 2) for p in p_adjusted]

    mean_z = np.mean(z_values)
    mean_p = 2 * (1 - norm.cdf(mean_z))
    mean_ks_stat = np.mean(ks_statistics)


    return mean_p, mean_ks_stat


def mann_whitney_u_test(data, feature, subsample_size=100, n_iterations=1000):
    group_0 = data[data['label'] == 0][feature].to_numpy()
    group_1 = data[data['label'] == 1][feature].to_numpy()

    p_values = []
    u_statistics = []
    z_scores = []


    for _ in range(n_iterations):
        # Randomly sample from the two groups
        sample_0 = np.random.choice(group_0, size=min(subsample_size, len(group_0)), replace=True)
        sample_1 = np.random.choice(group_1, size=min(subsample_size, len(group_1)), replace=True)

        # Perform Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(sample_0, sample_1, alternative='two-sided')
        u_statistics.append(u_stat)

        mean_u = len(sample_0) * len(sample_1) / 2
        std_u = np.sqrt(len(sample_0) * len(sample_1) * (len(sample_0) + len(sample_1) + 1) / 12)
        z_score = (u_stat - mean_u) / std_u
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        z_scores.append(z_score)

        p_values.append(p_value)

    # Calculate mean U statistic and mean p-value
    mean_u = np.mean(u_statistics)
    mean_p = np.mean(p_values)
    return mean_p, mean_u





feature_ot = pd.read_csv('/real_Pz.csv')
feature_mw = pd.read_csv('/Muse_Pz.csv')


feature_ot['label'] = 0
feature_mw['label'] = 1


data_combined = pd.concat([feature_ot, feature_mw])



feature_columns = [col for col in feature_ot.columns if col not in ['SampleID', 'label']]

# print(data_combined)



results = {}
#

for feature in feature_columns:
    mean_p_ks, mean_ks_stat = bootstrap_ks_test_Z(data_combined, feature)


    mean_p, mean_u = mann_whitney_u_test(data_combined, feature, n_iterations=100)

    results[feature] = {
        'Bootstrap KS p-value': mean_p_ks,
        'KS stas': mean_ks_stat,
        'Mann-Whitney U p-value': mean_p,
        'Mann-Whitney U stas': mean_u
    }

for feature, result in results.items():
    print(f"Feature: {feature}")
    print(f"  Bootstrap KS p-value, Stat: {result['Bootstrap KS p-value']:.4f}, {result['KS stas']:.4f}")
    print(f"  Mann-Whitney U p-value: {result['Mann-Whitney U p-value']:.4f}, {result['Mann-Whitney U stas']:.4f}\n")
