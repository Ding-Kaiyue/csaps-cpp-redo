#!/usr/bin/env python3
"""
验证 C++ CSAPS 实现与 Python 版本的一致性

用法：
    python3 verify_against_python.py
"""

import numpy as np
import sys
import json
from pathlib import Path

# 添加 Python csaps 到路径
sys.path.insert(0, str(Path.home() / 'Projects' / 'csaps'))

from csaps import csaps

def save_test_data(filename):
    """生成测试数据并保存为 JSON"""

    # 测试案例 1：简单的正弦曲线
    x1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
    y1 = np.sin(x1)

    # 测试案例 2：带权重的数据
    x2 = np.linspace(0, 10, 11)
    y2 = np.cos(x2)
    w2 = np.ones(11)
    w2[5] = 2.0  # 中间点权重更高

    # 测试案例 3：多维数据
    x3 = np.linspace(0, 1, 8)
    # Python csaps 期望 (n_dims, n_points) 的格式
    y3 = np.array([
        np.sin(2*np.pi*x3),
        np.cos(2*np.pi*x3),
        x3
    ])

    # 查询点
    xi = np.linspace(0, 5, 20)
    xi2 = np.linspace(0, 10, 25)
    xi3 = np.linspace(0, 1, 15)

    data = {
        "test_cases": [
            {
                "name": "case1_simple_sine",
                "x": x1.tolist(),
                "y": y1.tolist(),
                "weights": None,
                "smooth": None,  # auto
                "xi": xi.tolist(),
                "nu": 0,
                "extrapolate": True
            },
            {
                "name": "case2_weighted",
                "x": x2.tolist(),
                "y": y2.tolist(),
                "weights": w2.tolist(),
                "smooth": None,  # auto
                "xi": xi2.tolist(),
                "nu": 0,
                "extrapolate": True
            },
            {
                "name": "case3_manual_smooth",
                "x": x1.tolist(),
                "y": y1.tolist(),
                "weights": None,
                "smooth": 0.5,  # manual
                "xi": xi.tolist(),
                "nu": 0,
                "extrapolate": True
            },
            {
                "name": "case4_multivariate",
                "x": x3.tolist(),
                "y": y3.tolist(),
                "weights": None,
                "smooth": None,  # auto
                "xi": xi3.tolist(),
                "nu": 0,
                "extrapolate": True,
                "is_multivariate": True
            },
            {
                "name": "case5_derivative",
                "x": x1.tolist(),
                "y": y1.tolist(),
                "weights": None,
                "smooth": None,
                "xi": xi.tolist(),
                "nu": 1,  # first derivative
                "extrapolate": True
            },
            {
                "name": "case6_second_derivative",
                "x": x1.tolist(),
                "y": y1.tolist(),
                "weights": None,
                "smooth": None,
                "xi": xi.tolist(),
                "nu": 2,  # second derivative
                "extrapolate": True
            },
            {
                "name": "case7_no_extrapolate",
                "x": x1.tolist(),
                "y": y1.tolist(),
                "weights": None,
                "smooth": None,
                "xi": np.linspace(-1, 6, 20).tolist(),
                "nu": 0,
                "extrapolate": False
            }
        ]
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    return data

def compute_python_reference(data):
    """计算 Python csaps 的参考结果"""

    results = {}

    for test in data["test_cases"]:
        case_name = test["name"]
        x = np.array(test["x"])
        xi = np.array(test["xi"])

        if test.get("is_multivariate"):
            y = np.array(test["y"])
        else:
            y = np.array(test["y"])

        weights = np.array(test["weights"]) if test["weights"] else None
        smooth = test["smooth"]
        nu = test["nu"]
        extrapolate = test["extrapolate"]

        try:
            # 创建样条
            if test.get("is_multivariate"):
                sp = csaps(x, y, weights=weights, smooth=smooth)
                y_result = sp(xi, nu=nu, extrapolate=extrapolate)
                # 如果是多维，保留所有维度
                y_values = y_result.tolist() if y_result.ndim == 2 else [y_result.tolist()]
            else:
                sp = csaps(x, y, weights=weights, smooth=smooth)
                y_result = sp(xi, nu=nu, extrapolate=extrapolate)
                y_values = y_result.tolist()

            # 获取平滑因子
            if hasattr(sp, 'smooth'):
                smooth_used = float(sp.smooth) if np.isscalar(sp.smooth) else float(sp.smooth[0])
            else:
                smooth_used = None

            results[case_name] = {
                "y": y_values,
                "smooth": smooth_used,
                "success": True
            }

            print(f"✓ {case_name}: OK")

        except Exception as e:
            results[case_name] = {
                "error": str(e),
                "success": False
            }
            print(f"✗ {case_name}: ERROR - {e}")

    return results

def save_reference_results(filename, results):
    """保存参考结果"""
    # 包装在 test_results 中以匹配 C++ 的格式
    wrapped = {"test_results": results}
    with open(filename, 'w') as f:
        json.dump(wrapped, f, indent=2)

def main():
    print("=" * 70)
    print("CSAPS 验证：C++ vs Python")
    print("=" * 70)

    # 生成测试数据
    print("\n1. 生成测试数据...")
    test_data_file = Path("test_data.json")
    test_data = save_test_data(test_data_file)
    print(f"   ✓ 测试数据保存到 {test_data_file}")

    # 计算 Python 参考结果
    print("\n2. 计算 Python csaps 参考结果...")
    python_results = compute_python_reference(test_data)

    python_results_file = Path("python_reference_results.json")
    save_reference_results(python_results_file, python_results)
    print(f"   ✓ 参考结果保存到 {python_results_file}")

    # 统计
    success_count = sum(1 for r in python_results.values() if r.get("success", False))
    total_count = len(python_results)

    print("\n" + "=" * 70)
    print(f"Python 参考计算完成: {success_count}/{total_count} 成功")
    print("=" * 70)

    print("\n现在请运行 C++ 程序:")
    print(f"  cd /home/dyk/Projects/csaps-cpp")
    print(f"  ./verify_against_cpp test_data.json cpp_results.json")

    print("\n然后验证结果:")
    print(f"  python3 verify_against_python.py --compare")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # 对比模式
        print("\n对比 C++ 和 Python 结果...")
        try:
            with open("python_reference_results.json") as f:
                python_res = json.load(f)
            with open("cpp_results.json") as f:
                cpp_res = json.load(f)

            print("\n等待实现对比逻辑...")
        except FileNotFoundError as e:
            print(f"错误：找不到结果文件 - {e}")
    else:
        main()
