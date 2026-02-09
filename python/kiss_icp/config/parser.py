from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Import thư viện C++ binding
from kiss_icp.pybind import kiss_icp_pybind

from kiss_icp.config.config import (
    AdaptiveThresholdConfig,
    DataConfig,
    MappingConfig,
    RegistrationConfig,
)


class KISSConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="kiss_icp_")
    out_dir: str = "results"
    data: DataConfig = DataConfig()
    registration: RegistrationConfig = RegistrationConfig()
    mapping: MappingConfig = MappingConfig()
    adaptive_threshold: AdaptiveThresholdConfig = AdaptiveThresholdConfig()


def _yaml_source(config_file: Optional[Path]) -> Dict[str, Any]:
    data = None
    if config_file is not None:
        try:
            yaml = importlib.import_module("yaml")
        except ModuleNotFoundError:
            print(
                "Custom configuration file specified but PyYAML is not installed on your system,"
                ' run `pip install "kiss-icp[all]"`. You can also modify the config.py if your '
                "system does not support PyYaml "
            )
            sys.exit(1)
        with open(config_file) as cfg_file:
            data = yaml.safe_load(cfg_file)
    return data or {}


def load_config(config_file: Optional[Path]) -> KISSConfig:
    """Load configuration from an optional yaml file."""
    config = KISSConfig(**_yaml_source(config_file))

    # Check if there is a possible mistake
    if config.data.max_range < config.data.min_range:
        print("[WARNING] max_range is smaller than min_range, setting min_range to 0.0")
        config.data.min_range = 0.0

    # Use specified voxel size or compute one using the max range
    if config.mapping.voxel_size is None:
        config.mapping.voxel_size = float(config.data.max_range / 100.0)

    return config


def write_config(config: KISSConfig = KISSConfig(), filename: str = "kiss_icp.yaml"):
    with open(filename, "w") as outfile:
        try:
            yaml = importlib.import_module("yaml")
            yaml.dump(config.model_dump(), outfile, default_flow_style=False)
        except ModuleNotFoundError:
            outfile.write(str(config.model_dump()))


# --- HÀM MỚI (Giống to_genz_config) ---
def to_kiss_config(config: KISSConfig):
    """
    Chuyển đổi từ Pydantic Model (Python) sang Struct C++ (_KISSConfig).
    Cần mapping đúng các trường dữ liệu.
    """
    # 1. Khởi tạo struct C++
    cpp_config = kiss_icp_pybind._KISSConfig()
    
    # 2. Gán giá trị (Mapping từ Config lồng nhau sang Config phẳng của C++)
    
    # Data Config
    cpp_config.max_range = config.data.max_range
    cpp_config.min_range = config.data.min_range
    cpp_config.deskew = config.data.deskew
    
    # Mapping Config
    # Lưu ý: voxel_size lúc này chắc chắn đã có giá trị (do load_config xử lý rồi)
    cpp_config.voxel_size = config.mapping.voxel_size
    cpp_config.max_points_per_voxel = config.mapping.max_points_per_voxel
    
    # Adaptive Threshold Config
    cpp_config.min_motion_th = config.adaptive_threshold.min_motion_th
    cpp_config.initial_threshold = config.adaptive_threshold.initial_threshold
    
    # Registration Config
    cpp_config.max_num_iterations = config.registration.max_num_iterations
    cpp_config.convergence_criterion = config.registration.convergence_criterion
    cpp_config.max_num_threads = config.registration.max_num_threads

    return cpp_config
