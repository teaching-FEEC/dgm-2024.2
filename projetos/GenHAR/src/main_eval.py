import sys
import os
import argparse
from utils import log

REPO_ROOT_DIR = "../"
sys.path.append(os.path.dirname(REPO_ROOT_DIR))

from evals.evaluator import Evaluator

def main(results_dir, eval_dir):
    log.print_debug("START EVAL")
    evaluator = Evaluator(results_dir, eval_dir)
    evaluator.calculate_metrics()
    evaluator.create_reports()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Avaliação de resultados.')
    parser.add_argument('results_dir', type=str, help='Caminho para os resultados')
    parser.add_argument('eval_dir', type=str, help='Caminho para o diretório de avaliação')
    args = parser.parse_args()
    main(args.results_dir, args.eval_dir)
