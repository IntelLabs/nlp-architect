import numpy as np
from collections import defaultdict
from scipy import stats
import os

def t_test(filename_A, filename_B, alpha):
    with open(filename_A) as f:
        data_A = f.read().splitlines()

    with open(filename_B) as f:
        data_B = f.read().splitlines()

    data_A = list(map(float,data_A))
    data_B = list(map(float,data_B))

    # Paired Student's t-test: Calculate the T-test on TWO RELATED samples of scores, a and b. for one sided test we multiply p-value by half
    t_results = stats.ttest_rel(data_A, data_B)
    # correct for one sided test
    pval = float(t_results[1]) / 2
    if (float(pval) <= float(alpha)):
        # print("\nTest result is significant with p-value: {}".format(pval))
        return pval
    else:
        # print("\nTest result is not significant with p-value: {}".format(pval))
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
    # print "\n The K-Bonferroni estimator for the number of datasets with effect is: ", find_k_estimator(pvals, alpha, 'B')
    # print "\n The K-Fisher estimator for the number of datasets with effect is: ", find_k_estimator(pvals, alpha, 'F')
    k_est = find_k_estimator(pvals, alpha, 'B')
    rejlist = Holm(pvals, alpha)
    # print "\n The rejections list according to the Holm procedure is: "
    # for rej in rejlist:
    #     print "dataset"+str(rej+1)
    return k_est, rejlist


def main():
    root = '/Users/dkorat/Desktop/f1_drop_nulls/'
    datasets = ['dev_to_res', 'lap_to_res', 'res_to_lap', 'dev_to_lap', 'lap_to_dev', 'res_to_dev']
    seeds = ['12', '57', '89']
    splits = ['1', '2', '3']

    seed_pvals = defaultdict(list)

    for dataset in datasets:
        base_rnd = os.listdir(root + dataset + '/rnd')[2].split('sents')[0]
        base_libert = os.listdir(root + dataset + '/libert')[2].split('sents')[0]

        for seed in seeds:
            for split in splits:
                sample = 'sents_op_f1_seed_' + seed + '_split_' + split + '.txt'
                rnd = root + dataset + '/rnd/' + base_rnd + sample
                libert = root + dataset + '/libert/' + base_libert + sample
                p_val = t_test(rnd, libert, 0.1)
                sample_str = ' '.join([dataset, 'seed', seed, 'split', split, ':', str(p_val)])
                print(sample_str)
                seed_pvals[seed].append((p_val, sample_str))
        print()

    for alpha in 0.137, 0.15:
        all_pvals = []

        for seed, pvals_sample_strs in seed_pvals.items():
            pvals = [ps[0] for ps in pvals_sample_strs]
            sample_strs = [ps[1] for ps in pvals_sample_strs]
            k, rej_list = replicability(alpha, pvals)
            all_pvals.extend(pvals)
            print('The Bonferroni-k estimator for the number of datasets with effect is: \n', k, \
                '(out of ' + str(len(pvals)) + ')')
            rej_samples = '\n'.join([sample_strs[i] for i in rej_list])
            print('\nThe rejections list according to the Holm procedure is: \n', rej_samples, '\n')
            acc_samples = '\n'.join([sample_strs[i] for i in range(len(pvals)) if i not in rej_list])
            print('The acceptance list according to the Holm procedure is: \n', acc_samples, '\n')
            print()

        # print 'Replicability for all seeds together:'
        # k, rej_list = replicability(alpha, all_pvals)
        # print 'Number of datasets: ', len(all_pvals)
        # print 'The Bonferroni-k estimator for the number of datasets with effect is: ', k
        # print 'The rejections list according to the Holm procedure is: ', rej_list.tolist()
        # print 