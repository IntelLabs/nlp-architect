# pylint: disable=logging-fstring-interpolation
import numpy as np
from collections import defaultdict
from scipy import stats
from itertools import product
from pathlib import Path
from pytorch_lightning import _logger as log
from os import listdir

def t_test(filename_A, filename_B):
    with open(filename_A) as f:
        data_A = f.read().splitlines()
    with open(filename_B) as f:
        data_B = f.read().splitlines()
    data_A = list(map(float, data_A))
    data_B = list(map(float, data_B))

    # Paired Student's t-test: Calculate the T-test on TWO RELATED samples of scores, a and b. 
    # For one sided test we multiply p-value by half
    t_results = stats.ttest_rel(data_A, data_B)
    # correct for one sided test
    pval = float(t_results[1]) / 2
    return pval
        
def find_k_estimator(pvalues, alpha, method ='B'):
    """
    Calculates the estimator for k according to the method for combining the p-values given as input.

    Args:
        pvals: list of p-values.
        alpha: significance level.
        method: 'B' for Bonferroni  or 'F' for Fisher.
    Returns:
        k_hat: estimator of k calculated according to method.
    """
    n = len(pvalues)
    pc_vec = [1] * n
    k_hat = 0
    pvalues = sorted(pvalues, reverse = True)
    for u in range(0,n):
        if (u == 0):
            pc_vec[u] = calc_partial_cunjunction(pvalues, u+1, method)
        else:
            pc_vec[u] = max(calc_partial_cunjunction(pvalues, u+1, method), pc_vec[u - 1])
    k_hat = len([i for i in pc_vec if i<=alpha])
    return k_hat

def Holm(pvalues, alpha):
    """
    Applies the Holm procedure to determine the rejections list.

    Args:
        pvals: list of p-values.
        alpha: significance level.
    Returns:
        list of indeces of rejected null hypotheses. The rejected hypotheses are the hatk_Bonf. smallest p-values
    """
    k = find_k_estimator(pvalues, alpha)
    A = np.array(pvalues)
    idx = np.argpartition(A, k-1)
    return idx[:k]

def calc_partial_cunjunction(pvalues, u, method ='B'):
    """
    This function calculates the partial conjunction p-value of u out of n.

    Args:
        pvals: list sorted of p-values from big to small.
        u: number of hypothesized true null hypotheses.
        method: 'B' for Bonferroni  or 'F' for Fisher.
    Returns:
        p_u_n: p-value for the partial conjunction hypothesis of u out of n.
    """
    n = len(pvalues)
    sorted_pvlas = pvalues[0:(n-u+1)]
    if (method == 'B'):
        p_u_n = (n-u+1) * min(sorted_pvlas)
    elif (method == 'F'):
        sum_chi_stat = 0
        for p in sorted_pvlas:
            sum_chi_stat = sum_chi_stat -2 * np.log(p)
        p_u_n = 1-stats.chi2.cdf(sum_chi_stat, 2 * (n-u+1))
    return p_u_n

def replicability(alpha, pvals):
    # Calculate the K-Bonferroni estimator for the number of datasets with effect
    k_est = find_k_estimator(pvals, alpha, 'B')
    # Get the rejections list according to the Holm procedure
    rejlist = Holm(pvals, alpha)
    return k_est, rejlist


def significance_report(
        datasets: list,
        version: str, 
        seeds: list,
        splits: list,
        log_root: Path,
        model: str = 'libert',
        baseline: str = 'libert_random_init',
        epochs: int = 1,
        alphas: tuple = (.001, .01, .05, .1, .15, .2, .3)):

    res_str = ""
    for epoch in range(epochs):
        seed_pvals = defaultdict(list)
        for data, seed, split in product(datasets, seeds, splits):
            baseline_txt = log_root / data / f'{baseline}_seed_{seed}_split_{split}' / 'version_0' / 'tf' / f'sent_f1_epoch_{epoch}.txt'
            model_txt = log_root / data / f'{model}_seed_{seed}_split_{split}' / version / 'tf' / f'sent_f1_epoch_{epoch}.txt'

            p_val = t_test(baseline_txt, model_txt)
            sample_str = f'{data} seed_{seed} split_{split} epoch_{epoch}: {p_val}'
            seed_pvals[seed].append((p_val, sample_str))

        for alpha in alphas:
            res_str += f"\n\n{'=' * 40}\nAlpha (p-value): {alpha}\n{'=' * 40}\n"

            all_pvals = []
            scores = []
            for seed, pvals_sample_strs in seed_pvals.items():
                res_str += f"Seed: {seed}\n{'-' * 10}\n"
                pvals = [ps[0] for ps in pvals_sample_strs]
                sample_strs = [ps[1] for ps in pvals_sample_strs]
                k, rej_list = replicability(alpha, pvals)

                score = k / float(len(pvals))
                scores.append(score)
                res_str += f"Score: {score}\n"

                all_pvals.extend(pvals)
                res_str += f"The Bonferroni-k estimator for the number of datasets with effect is: {k} (out of {len(pvals)})\n"

                rej_samples = '\n'.join([sample_strs[i] for i in rej_list])
                res_str += f"\nThe rejections list according to the Holm procedure is:\n{rej_samples}\n\n"

                acc_samples = '\n'.join([sample_strs[i] for i in range(len(pvals)) if i not in rej_list])
                res_str += f"The acceptance list according to the Holm procedure is:\n{acc_samples}\n\n"

            res_str += "---------------------------------------------------\n"
            res_str += f"Avg. score with alpha = {alpha}:\n{np.mean(scores)}\n"
            res_str += "---------------------------------------------------\n"

            res_str += "\n\n+++++++++++++++++++++++++++++++++++++\n"
            res_str += "Replicability for all seeds together:'\n"
            k, rej_list = replicability(alpha, all_pvals)
            res_str += f"Number of datasets: {len(all_pvals)}\n"
            res_str += f"The Bonferroni-k estimator for the number of datasets with effect is: {k}\n"
            res_str += f"The rejections list according to the Holm procedure is: {rej_list.tolist()}\n"
            res_str += f"Score: {k / float(len(all_pvals))}\n"

    log.info(res_str)

    template = lambda i: f"significance_{model}_vs_{baseline}_{i}.txt"
    i = 0
    while template(i) in listdir(log_root):
        i += 1
    with open(log_root / template(i), 'w') as f:
        f.write(res_str)
