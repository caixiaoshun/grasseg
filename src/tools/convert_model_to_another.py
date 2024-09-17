import os
import argparse
from typing import Tuple
from glob import glob

def get_parser()->Tuple[str,str]:
    parser = argparse.ArgumentParser(description='Convert model config to another model')
    parser.add_argument('--model1', type=str, required=True, help='a model name')
    parser.add_argument('--model2', type=str, required=True, help='another model name')

    args = parser.parse_args()
    model1 = args.model1
    model2 = args.model2
    print(f'convert {model1} to {model2}')
    return model1, model2

def main():
    model1, model2 = get_parser()
    model1_configs = glob(f'configs/experiment/*{model1}*.yaml')
    script_path = glob(f"scripts/train-{model1}.sh")

    assert len(model1_configs) > 0, f'No config found for {model1}'
    assert len(script_path) == 1, f'No script found for {model1}'

    script_path = script_path[0]
    print(f'Found {len(model1_configs)} configs for {model1}')

    for config in model1_configs:
        content = open(config, 'r').read()
        content = content.replace(model1, model2)
        new_config = config.replace(model1, model2)
        with open(new_config, 'w') as f:
            f.write(content)
    print(f'Converted {len(model1_configs)} configs from {model1} to {model2}')
    content = open(script_path, 'r').read()
    content = content.replace(model1, model2)
    new_script_path = script_path.replace(model1, model2)
    with open(new_script_path, 'w') as f:
        f.write(content)
    print(f'Converted script from {model1} to {model2}')



if __name__ == '__main__':
    #example python src/tools/convert_model_to_another.py --model1 pspnet --model2 unet
    main()