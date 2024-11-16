"""Build a workflow diagram."""
from pathlib import Path
from graphviz import Digraph

def create_workflow_diagram(file_path):
    """Build a workflow diagram and save it to a file."""
    g = Digraph(comment='CycleGAN Workflow')
    g.attr(rankdir='LR', fontsize='12', fontname='Arial')

    # Define color scheme
    process_color = 'lightblue'
    data_color = 'lightgrey'
    auxiliary_color = 'lightyellow'
    output_color = 'palegreen'

    # Input and Preprocessing
    g.node('input_images', label='Input Images\n(Domain A & B)\nFormats: PNG/JPEG\nDimensions: 256x256x3', style='filled', shape='cylinder', fillcolor=data_color)
    g.node('preprocessing', label='Preprocessing\n- Resize\n- Normalize [-1, 1]', shape='box', style='filled', fillcolor=process_color)
    g.edge('input_images', 'preprocessing', label='Raw Images')

    # Training Phase
    g.node('train_data', label='Training Data\n80% Preprocessed Images', style='filled', shape='cylinder', fillcolor=data_color)
    g.node('training', label='Training CycleGAN\nOutputs:\n- Generators (gen_AtoB, gen_BtoA)\n- Discriminators (dis_A, dis_B)\nFiles: .pth', shape='box', style='filled', fillcolor=process_color)
    g.edge('preprocessing', 'train_data', label='Processed Data')
    g.edge('train_data', 'training', label='CycleGAN Training')

    # Inference Phase
    g.node('test_data', label='Testing Data\n20% Preprocessed Images', style='filled', shape='cylinder', fillcolor=data_color)
    g.node('inference', label='Inference\nGenerates Synthetic Images\nFormat: PNG, 256x256x3', shape='box', style='filled', fillcolor=process_color)
    g.edge('preprocessing', 'test_data', label='Processed Test Data')
    g.edge('test_data', 'inference', label='Synthetic Image Generation')
    g.edge('training', 'inference', label='Trained Models (.pth)')

    # Metrics Calculation
    g.node('metrics_calc', label='Metrics Calculation\n- FID\n- LPIPS\nCompares Real vs Synthetic', shape='box', style='filled', fillcolor=process_color)
    g.node('fid', label='FID Score\nEvaluates Distribution Similarity', style='filled', shape='ellipse', fillcolor=output_color)
    g.node('lpips', label='LPIPS Score\nEvaluates Perceptual Similarity', style='filled', shape='ellipse', fillcolor=output_color)
    g.edge('inference', 'metrics_calc', label='Synthetic Images')
    g.edge('test_data', 'metrics_calc', label='Real Images')
    g.edge('metrics_calc', 'fid', label='FID Output')
    g.edge('metrics_calc', 'lpips', label='LPIPS Output')

    # Auxiliary Components
    g.node('replay_buffer', label='Replay Buffer\nSize: 50 Images\nReduces Discriminator Oscillation', shape='ellipse', style='filled', fillcolor=auxiliary_color)
    g.node('losses', label='Loss Functions\n- Adversarial\n- Cycle Consistency\n- Identity\n- Path Length Penalty', shape='ellipse', style='filled', fillcolor=auxiliary_color)
    g.edge('training', 'replay_buffer', label='Buffer Usage')
    g.edge('training', 'losses', label='Loss Calculations')

    # Rendering
    g.render(file_path, format='png')

def main():
    """Create a workflow diagram."""
    file_name = 'workflow'
    create_workflow_diagram(Path(__file__).parent / 'assets' / file_name)

if __name__ == '__main__':
    main()
