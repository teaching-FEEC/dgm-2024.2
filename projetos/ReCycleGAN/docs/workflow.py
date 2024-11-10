"""Build a workflow diagram."""
from pathlib import Path
from graphviz import Digraph

def create_workflow_diagram(file_path):
    """Build a workflow diagram and save it to a file."""
    g = Digraph(comment='Workflow')
    g.attr(rankdir='LR')

    g.node('imagens', label='Imagens\nReais', style='filled', shape='cylinder')
    g.node('lpips', label='LPIPS', style='filled')
    g.node('fid', label='FID', style='filled')

    g.node('treino', label='Treino do\nModelo', shape='box')
    g.node('inferencia', label='Inferência com\no Modelo', shape='box')
    g.node('calc_metricas', label='Cálculo\nde Métricas', shape='box')

    g.edge('imagens', 'treino', label='Reais\ntreino')
    g.edge('treino', 'inferencia', label='Modelo\ntreinado')
    g.edge('imagens', 'inferencia', label='Reais\ntreino+teste')
    g.edge('imagens', 'calc_metricas', label='Reais\nteste+treino')
    g.edge('inferencia', 'calc_metricas', label='Transformadas\nteste+treino')
    g.edge('calc_metricas', 'fid')
    g.edge('calc_metricas', 'lpips')

    g.render(file_path, format='png')

def main():
    """Create a workflow diagram."""
    file_name = 'workflow'
    create_workflow_diagram(Path(__file__).parent / 'assets' / file_name)

if __name__ == '__main__':
    main()
