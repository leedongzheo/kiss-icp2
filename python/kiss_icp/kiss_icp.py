# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable

import numpy as np

# Import thư viện pybind đã biên dịch
from kiss_icp.pybind import kiss_icp_pybind


@dataclass
class KISSConfig:
    # Các tham số mặc định lấy từ KISSConfig struct cũ
    voxel_size: float = 1.0
    max_range: float = 100.0
    min_range: float = 0.0
    max_points_per_voxel: int = 20
    min_motion_th: float = 0.1
    initial_threshold: float = 2.0
    max_num_iterations: int = 500
    convergence_criterion: float = 0.0001
    max_num_threads: int = 0
    deskew: bool = True

    def _to_cpp(self):
        """Chuyển đổi Config Python sang C++ Struct"""
        config = kiss_icp_pybind._KISSConfig()
        config.voxel_size = self.voxel_size
        config.max_range = self.max_range
        config.min_range = self.min_range
        config.max_points_per_voxel = self.max_points_per_voxel
        config.min_motion_th = self.min_motion_th
        config.initial_threshold = self.initial_threshold
        config.max_num_iterations = self.max_num_iterations
        config.convergence_criterion = self.convergence_criterion
        config.max_num_threads = self.max_num_threads
        config.deskew = self.deskew
        return config


def _to_cpp_points(frame: np.ndarray):
    """Helper: Chuyển Numpy Array sang C++ Vector3d"""
    points = np.asarray(frame, dtype=np.float64)
    return kiss_icp_pybind._Vector3dVector(points)


class KissICP:
    def __init__(self, config: Optional[KISSConfig] = None):
        # 1. Quản lý Config
        self.config = config or KISSConfig()
        
        # 2. Khởi tạo "Hộp đen" C++ (Monolithic Wrapper)
        # Toàn bộ logic Preprocess -> Voxelize -> Register -> Map Update nằm trong này
        self._pipeline = kiss_icp_pybind._KissICP(self.config._to_cpp())

    def register_frame(
        self, frame: np.ndarray, timestamps: Optional[Iterable[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Input: Raw Point Cloud (Numpy)
        Output: Tuple (Deskewed Frame, Keypoints Source)
        """
        points = _to_cpp_points(frame)
        
        # Xử lý timestamp (nếu không có thì truyền list rỗng)
        ts_list = list(timestamps) if timestamps is not None else []

        # Gọi xuống C++
        # Hàm C++ trả về std::tuple<Vector, Vector> -> Python nhận là tuple(list, list)
        frame_out, source_out = self._pipeline._register_frame(points, ts_list)
        
        return np.asarray(frame_out), np.asarray(source_out)

    @property
    def last_pose(self) -> np.ndarray:
        """Trả về pose cuối cùng (4x4 Matrix) từ C++"""
        return np.asarray(self._pipeline._last_pose())

    @property
    def local_map(self) -> np.ndarray:
        """Lấy toàn bộ điểm trong bản đồ hiện tại"""
        return np.asarray(self._pipeline._local_map())

    def reset(self):
        """Reset thuật toán về trạng thái ban đầu"""
        self._pipeline._reset()

# Hàm tiện ích (Global function) giống GenZ
def voxel_down_sample(frame: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.asarray(kiss_icp_pybind._voxel_down_sample(_to_cpp_points(frame), voxel_size))
