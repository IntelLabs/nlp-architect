import sys
from run_ray import read_config, prepare_config, \
    post_analysis, prepare_all_formalisms_summary_table, LOG_ROOT

def post_analysis_for_debug(config_yaml, exp_id):
    cfg = read_config(config_yaml)
    for formalism in cfg.formalisms.split():
        cfg.formalism = formalism
        prepare_config(cfg)
        log_dir = LOG_ROOT / exp_id / cfg.formalism
        post_analysis(cfg, log_dir, exp_id)
    prepare_all_formalisms_summary_table(exp_id)

# if __name__=="__main__":
#     post_analysis_for_debug(*sys.argv[1:])