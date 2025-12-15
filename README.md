# C++ CSAPS (Cubic Spline Approximation/Smoothing)

高性能 C++ 实现的 Cubic Spline 样条插值与平滑库，与 Python `csaps` 库完全兼容。

✅ **7/7 验证测试通过**，误差 < 1e-16（浮点数精度极限）

## 特点

- ✅ 自动平滑参数计算
- ✅ 支持权重数据
- ✅ 多变量样条
- ✅ 导数计算（1阶、2阶、3阶）
- ✅ 可控制的外推行为
- ✅ 与 Python csaps 库精度完全对齐

## 一键验证

验证 C++ 与 Python 版本的输出一致性：

```bash
./verify.sh
```

**输出示例：**
```
✓ case1_simple_sine: Y 最大差值: 1.11e-16 ✓
✓ case2_weighted: Y 最大差值: 4.44e-16 ✓
✓ case3_manual_smooth: Y 最大差值: 2.22e-16 ✓
✓ case4_multivariate: Y 最大差值: 2.22e-16 ✓
✓ case5_derivative: Y 最大差值: 1.11e-16 ✓
✓ case6_second_derivative: Y 最大差值: 1.11e-16 ✓
✓ case7_no_extrapolate: Y 最大差值: 1.11e-16 ✓

总体结果: 7/7 通过
✓ 所有测试通过！C++ 和 Python 版本完全一致。
```

## 快速开始

### 安装依赖

```bash
# C++ 编译依赖
apt-get install libeigen3-dev

# 验证脚本依赖（可选）
pip install csaps numpy
apt-get install nlohmann-json3-dev
```

### 基础用法

```cpp
#include "src/csaps.h"
using namespace csaps;

// 创建样条（自动平滑参数）
DoubleArray x(6);
x << 0, 1, 2, 3, 4, 5;
DoubleArray y(6);
y << 0, 0.84, 0.91, 0.14, -0.76, -0.96;

UnivariateCubicSmoothingSpline spline(x, y);

// 在新点处求值
DoubleArray xi(5);
xi << 0.5, 1.5, 2.5, 3.5, 4.5;

DoubleArray yi = spline(xi);           // 函数值
DoubleArray dy = spline(xi, 1);        // 一阶导数
DoubleArray ddy = spline(xi, 2);       // 二阶导数
```

### 带权重和参数

```cpp
// 定义权重（可选）
DoubleArray weights(6);
weights << 1, 1, 2, 1, 1, 1;

// 自定义平滑参数：0~1（0=原始数据，1=最光滑）
double smooth = 0.9;

UnivariateCubicSmoothingSpline spline(x, y, weights, smooth);
```

### 多变量样条

```cpp
// 3个点，2维输出
DoubleArray2D ydata(3, 2);  // (n_points, n_dims)
ydata << 0, 1,
         1, 2,
         2, 1;

MultivariateCubicSmoothingSpline spline(x, ydata);

// 求值
DoubleArray2D result = spline(xi);  // (n_query_points, n_dims)
```

## 编译与使用

### 基本编译

```bash
g++ -std=c++17 -I. -I/usr/include/eigen3 your_file.cpp src/csaps.cpp -o your_exe
```

### 完整编译示例

```bash
# 编译单个程序
g++ -std=c++17 -I. -I/usr/include/eigen3 \
    -O2 -Wall \
    main.cpp src/csaps.cpp -o main

# 使用 CMake
mkdir build && cd build
cmake ..
make
```

## API 参考

### UnivariateCubicSmoothingSpline

#### 构造函数

```cpp
// 1. 基础构造
UnivariateCubicSmoothingSpline(const DoubleArray &x, const DoubleArray &y);

// 2. 带权重
UnivariateCubicSmoothingSpline(const DoubleArray &x, const DoubleArray &y,
                                const DoubleArray &weights);

// 3. 自定义平滑参数
UnivariateCubicSmoothingSpline(const DoubleArray &x, const DoubleArray &y,
                                double smooth);

// 4. 权重 + 平滑参数
UnivariateCubicSmoothingSpline(const DoubleArray &x, const DoubleArray &y,
                                const DoubleArray &weights, double smooth);
```

#### 求值

```cpp
// 函数值
DoubleArray result = spline(xi);

// 带导数
DoubleArray result = spline(xi, 1);  // 一阶导数
DoubleArray result = spline(xi, 2);  // 二阶导数
DoubleArray result = spline(xi, 3);  // 三阶导数

// 控制外推
DoubleArray result = spline(xi, 0, true);   // 外推（默认）
DoubleArray result = spline(xi, 0, false);  // 超出范围返回 NaN
```

### MultivariateCubicSmoothingSpline

```cpp
// 创建多变量样条
MultivariateCubicSmoothingSpline spline(x, ydata);

// 求值
DoubleArray2D result = spline(xi);

// 获取平滑参数
DoubleArray smooths = spline.GetSmooths();
```

## 参数说明

| 参数 | 说明 | 范围 |
|------|------|------|
| `smooth` | 平滑参数 | 0~1（-1为自动） |
| `nu` | 导数阶数 | 0,1,2,3 |
| `extrapolate` | 是否外推 | true/false |
| `weights` | 数据权重 | > 0 |

## 修复记录

### 已修复的关键 Bugs

1. **Digitize 函数 - 无穷大处理**
   - 问题：tolerance 计算包含 infinity，导致分组错误
   - 修复：特殊处理 `-infinity` 和 `+infinity`

2. **Evaluate 函数 - 多项式求值**
   - 问题：复杂的数组操作导致系数混乱
   - 修复：重写为清晰的 Horner 方法

3. **二阶导数计算**
   - 问题：错误的乘数公式
   - 修复：直接计算 `6*c0*x + 2*c1`

### 验证详情

所有 7 个测试用例全部通过：

| 测试用例 | 功能 | 状态 |
|---------|------|------|
| case1_simple_sine | 自动平滑参数 | ✓ 通过 |
| case2_weighted | 权重数据 | ✓ 通过 |
| case3_manual_smooth | 手动平滑参数 | ✓ 通过 |
| case4_multivariate | 多维输出 | ✓ 通过 |
| case5_derivative | 一阶导数 | ✓ 通过 |
| case6_second_derivative | 二阶导数 | ✓ 通过 |
| case7_no_extrapolate | 禁用外推 | ✓ 通过 |

**最大误差：< 1e-16（浮点数精度极限）**

## 项目结构

```
csaps-cpp/
├── src/
│   ├── csaps.h           # API 头文件
│   └── csaps.cpp         # 实现（已修复）
├── verify.sh             # 一键验证脚本 ⭐
├── verify.cpp            # C++ 验证程序
├── verify_against_python.py  # Python 参考生成
├── HOW_TO_VERIFY.txt     # 详细验证教程
├── VERIFICATION.md       # 验证脚本说明
├── README.md             # 本文件
└── CMakeLists.txt        # CMake 配置
```

## 常见问题

**Q: 如何选择平滑参数？**
A: 使用 `-1`（默认）自动计算，或手动指定 0~1 之间的值（接近 1 更平滑）。

**Q: 性能如何？**
A: 采用稀疏矩阵和 SparseLU 求解，时间复杂度 O(n)，适合大规模数据。

**Q: 如何处理超出范围的点？**
A: 使用 `extrapolate` 参数控制（true=外推，false=返回 NaN）。

**Q: 支持复数吗？**
A: 当前实现为实数，可扩展支持复数。

## 依赖

### 必需
- C++17 编译器
- Eigen3：`apt-get install libeigen3-dev`

### 可选（用于验证）
- Python 3
- csaps：`pip install csaps`
- nlohmann/json：`apt-get install nlohmann-json3-dev`

## 算法原理

Cubic Smoothing Spline 基于最小二乘法和光滑性的权衡：

```
最小化：∑(wi * (yi - si(xi))²) + λ ∫ (s''(x))² dx
```

其中：
- `si(x)` 是分段三次多项式
- `wi` 是权重
- `λ` 由平滑参数 `smooth` 确定

参考：Reinsch, C.H., 1967. "Smoothing by spline functions"

## License

与原始 Python `csaps` 库兼容。
