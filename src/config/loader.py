import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, get_args, get_origin

import yaml
from dotenv import load_dotenv


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def dict_to_dataclass(data: Dict[str, Any], target_class) -> Any:
    if not hasattr(target_class, '__dataclass_fields__'):
        return data
    
    kwargs = {}
    for field_name, field_type in target_class.__dataclass_fields__.items():
        if field_name in data:
            field_value = data[field_name]
            field_class = field_type.type
            
            if hasattr(field_class, '__dataclass_fields__'):
                kwargs[field_name] = dict_to_dataclass(field_value, field_class)
                continue
            
            origin = get_origin(field_class)
            args = get_args(field_class)
            if origin is list and args:
                item_class = args[0]
                if hasattr(item_class, '__dataclass_fields__') and isinstance(field_value, list):
                    kwargs[field_name] = [dict_to_dataclass(item, item_class) for item in field_value]
                else:
                    kwargs[field_name] = field_value
                continue

            kwargs[field_name] = field_value
    
    return target_class(**kwargs)


def load_config(
    config_name: str = 'default',
    config_dir: str = 'configs'
):
    load_dotenv(override=True)
    
    config_path = f'{config_dir}/{config_name}.yaml'
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    
    config_dict = load_yaml_config(config_path)
    
    model_name = config_dict['model']['name']
    tag = config_dict['model']['tag']
    
    if config_dict['general']['work_dir'] == 'default':
        config_dict['general']['work_dir'] = (
            f"{Path(config_dir).parent.absolute()}/outputs/hubert_mamba/{model_name}/{tag}"
        )
    elif config_dict['general']['work_dir'] == 'local':
        config_dict['general']['work_dir'] = f"{os.getcwd()}/{tag}"
    
    if 'testing_ckpt' in config_dict['general']:
        if config_dict['general']['testing_ckpt'] == 'default':
            config_dict['general']['testing_ckpt'] = f"{config_dict['general']['work_dir']}/checkpoint.pt"
        elif config_dict['general']['testing_ckpt'] == 'same':
            config_dict['general']['testing_ckpt'] = config_dict['general']['ckpt']['path']
    
    if 'HubertMamba' in model_name:
        from config.hubert_mamba import HubertMambaConfig
        
        default_config = HubertMambaConfig()
        default_dict = asdict(default_config)
        
        merged_config = merge_configs(default_dict, config_dict)
        return dict_to_dataclass(merged_config, HubertMambaConfig)

    raise ValueError(f"Unsupported model config: {model_name}. This repo now supports HuBERT-Mamba only.")


def config_to_yaml(config) -> str:
    config_dict = asdict(config)
    return yaml.dump(config_dict, default_flow_style=False, indent=2, allow_unicode=True, sort_keys=False)
