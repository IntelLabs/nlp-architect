import significance
from pathlib import Path

if __name__ == "__main__":
    args1 = {'datasets': ['laptops_to_restaurants', 'restaurants_to_laptops'],
         'exp_id': 'Sat_Aug_08_04:44:57',
         'seeds': ['12', '57', '89'],
         'splits':['1', '2', '3'],
         'log_root':  Path('/workdisk/projects/2020/absa_k/ww31/nlp-architect/nlp_architect/models/libert/logs/Sat_Aug_08_04:44:57'),
         'baseline': 'libert_baseline',
         'baseline_ver': 'version_baseline',
         'model': 'libert',
         'alphas': (.001, .01, .05, .1, .15, .2, .3)
        }
    significance.significance_report(**args1)