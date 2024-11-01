# pylint: disable=import-error,wrong-import-position
"""Script to load and evaluate models."""
import sys
from pathlib import Path
from test_model import translate_images

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.test_cases import TEST_CASES


def build_images(case):
    """Generates translated images the CycleGAN model."""
    print(f"Test Case {case}")
    params = TEST_CASES[str(case)]

    base_folder = Path(__file__).resolve().parent.parent.parent
    restart_folder = base_folder / f'data/checkpoints/test_case_{case}'
    restart_file = list(restart_folder.glob('*.pth'))
    if len(restart_file) == 0:
        print(f"Np pth file not found in: {restart_folder}")
        return
    if len(restart_file) > 1:
        print(f"Multiple pth files found in: {restart_folder}")
        return

    params['restart_path'] = restart_file[0]
    params['data_folder'] = base_folder / 'data/external/nexet'
    params['output_name'] = f'test_{case}'
    params['csv_type'] = ''
    params['device'] = 'cuda'
    # params['params_path'] = restart_file[0].with_suffix('.json')
    translate_images(params)


if __name__ == '__main__':

    for i in range(2, 4):
        build_images(i)
