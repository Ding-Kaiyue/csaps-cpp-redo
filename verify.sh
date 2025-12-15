#!/bin/bash

set -e

echo "================================================================================"
echo "C++ CSAPS vs Python CSAPS 验证脚本"
echo "================================================================================"
echo ""

# 检查依赖
echo "【步骤 1】检查依赖..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

if ! python3 -c "import csaps" 2>/dev/null; then
    echo "❌ csaps 库未安装，运行: pip install csaps"
    exit 1
fi

if [ ! -d "/usr/include/eigen3" ]; then
    echo "⚠️  Eigen3 未找到，尝试安装: apt-get install libeigen3-dev"
    exit 1
fi

echo "✓ 所有依赖已满足"
echo ""

# 生成测试数据和Python参考结果
echo "【步骤 2】生成测试数据和Python参考结果..."
python3 verify_against_python.py > /dev/null 2>&1
if [ -f "test_data.json" ] && [ -f "python_reference_results.json" ]; then
    echo "✓ 测试数据已生成"
else
    echo "❌ 生成测试数据失败"
    exit 1
fi
echo ""

# 编译C++验证程序
echo "【步骤 3】编译C++验证程序..."
if [ ! -f "verify.cpp" ]; then
    echo "❌ verify.cpp 不存在"
    exit 1
fi

if g++ -std=c++17 -I. -I/usr/include/eigen3 -I/usr/include verify.cpp src/csaps.cpp -o verify 2>/dev/null; then
    echo "✓ C++验证程序编译成功"
else
    echo "❌ C++编译失败"
    exit 1
fi
echo ""

# 运行C++验证
echo "【步骤 4】运行C++验证程序..."
if ./verify test_data.json cpp_results.json > /dev/null 2>&1; then
    echo "✓ C++程序运行成功"
else
    echo "❌ C++程序运行失败"
    exit 1
fi
echo ""

# 对比结果
echo "【步骤 5】对比 C++ 和 Python 结果..."
echo ""

python3 << 'PYTHON_SCRIPT'
import json
import numpy as np
import sys

try:
    with open('python_reference_results.json') as f:
        python_results = json.load(f)['test_results']
    with open('cpp_results.json') as f:
        cpp_results = json.load(f)['test_results']
except Exception as e:
    print(f"❌ 读取结果文件失败: {e}")
    sys.exit(1)

cases = list(python_results.keys())
passed = 0
failed = 0

print("=" * 80)
print("对比结果详情")
print("=" * 80)
print("")

for case in cases:
    if case not in cpp_results:
        print(f"❌ {case}: C++未运行")
        failed += 1
        continue

    py_res = python_results[case]
    cpp_res = cpp_results[case]

    if not cpp_res.get('success'):
        print(f"❌ {case}: {cpp_res.get('error', 'Failed')}")
        failed += 1
        continue

    # 比较 smooth 参数
    smooth_match = abs(py_res['smooth'] - cpp_res['smooth']) < 1e-14

    # 比较 Y 值
    py_y = np.array(py_res['y'], dtype=float)
    cpp_y = np.array(cpp_res['y'], dtype=float)

    # 处理 NaN 值
    valid_mask = ~(np.isnan(py_y) | np.isnan(cpp_y))

    if np.any(np.isnan(py_y) | np.isnan(cpp_y)):
        if np.sum(np.isnan(py_y)) != np.sum(np.isnan(cpp_y)):
            print(f"⚠️  {case}: NaN 个数不匹配 (Python: {np.sum(np.isnan(py_y))}, C++: {np.sum(np.isnan(cpp_y))})")
            failed += 1
            continue

    if np.any(valid_mask):
        valid_py = py_y[valid_mask]
        valid_cpp = cpp_y[valid_mask]
        max_diff = np.max(np.abs(valid_py - valid_cpp))
        mean_diff = np.mean(np.abs(valid_py - valid_cpp))

        # 判断通过/失败
        if max_diff < 1e-10:
            status = "✓"
        elif max_diff < 1e-6:
            status = "⚠️ "
        else:
            status = "❌"

        if status == "✓":
            passed += 1
        else:
            failed += 1

        print(f"{status} {case}")
        print(f"   smooth: py={py_res['smooth']:.15f}, cpp={cpp_res['smooth']:.15f} {'✓' if smooth_match else '❌'}")
        print(f"   Y 最大差值: {max_diff:.2e}")
        print(f"   Y 平均差值: {mean_diff:.2e}")
    else:
        print(f"❌ {case}: 没有有效数据")
        failed += 1

    print("")

print("=" * 80)
print(f"总体结果: {passed}/{len(cases)} 通过")
print("=" * 80)

if failed == 0:
    print("✓ 所有测试通过！C++ 和 Python 版本完全一致。")
    sys.exit(0)
else:
    print(f"❌ {failed} 个测试失败")
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 验证完成：C++ 与 Python 输出一致！"
    exit 0
else
    echo ""
    echo "❌ 验证失败"
    exit 1
fi
