# 基于transformer架构多头机制的大模型微调

华东理工大学 曾子航 朱颖慧 靳奇昂

## 项目简介

本项目实现了一个统一处理多种题型的Transformer模型，支持：

- 📝 **选择题**（分类任务）
- 💻 **代码生成题**（自回归生成）
- ✍️ **开放生成题**（文本生成）
- ➗ **数学题**（数值推理）

## 技术亮点

![75449003014](https://github.com/2019598399/-transformer-/blob/main/%E9%A1%B9%E7%9B%AE%E6%9E%B6%E6%9E%84%E8%AF%B4%E6%98%8E.png)

1. **多任务学习架构**
   - 共享Transformer编码器
   - 选择题与其他题型各有一个输出头
   - 虽然在本项目中没有特别大的用处，但是可以为多模态大模型微调任务提供一些思路
2. **高效微调方案**
   - 采用LoRA技术
   - 仅微调少量参数（<1%）
   - 大幅降低计算成本
3. **统一解决方案**
   - 单一模型处理所有题型
   - 减少部署复杂度

## 快速开始

### 安装依赖库

1.安装训练所需第三方库

```shell
pip install -r requirements.txt
```

训练支持Windows和Linux操作系统。

2.通过百度网盘下载原模型和合并参数后的模型，如果您需要自己训练，只需下载原模型：

Qwen2.5-3B(原模型):链接: https://pan.baidu.com/s/19xrCEUf_2ZBVzUnxEyRdTw?pwd=3s6g 提取码: 3s6g   
qwen2.5-merged-3b(合并参数后的模型):链接: https://pan.baidu.com/s/1O55B_f77IYDBwJ6y3egtFg 提取码: 8br3  
下载压缩包后在本地解压，将整个名为Qwen2.5-3B或qwen2.5-merged-3b文件夹移动到您的项目文件夹根目录下  

3.安装推理后端框架

```shell
mkdir -p requirements 
cd requirements
git clone --depth 1 --branch v0.8.4 https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e . --user
cd ..
```

vllm后端推理框架只支持linux操作系统。Windows操作系统可使用双系统或虚拟机进行部署。

虚拟机安装教程：[VMware 安装配置 Ubuntu（最新版、超详细）_vmware-workstation-full-17.5.1-23298084.exe-CSDN博客](https://blog.csdn.net/m0_70885101/article/details/137694608)

### 如何运行项目

```shell
python train.py
```

首先运行训练脚本。如果您没有微调所需计算资源，我已将训练好的参数合并进文件夹qwen2.5-3b-merged，您只需运行推理框架查看微调结果。

```shell
python inference.py
```

运行推理脚本。脚本默认打印前十条，您可自行修改。

## 项目文件结构

```shell
project/
├── train.py                # 训练脚本
├── model.py                # 模型脚本，内含多头前向传播路由与计算损失函数并返回损失值
├── inference.py            # 推理脚本
├── device.py               # 获取当前设备
├── requirements.txt        # 项目所需Python库和版本
├── Qwen2.5-3B/             # 未经微调的原始模型
├── qwen2.5-3b-merged/      # 已微调并合并参数的模型
├── train.jsonl             # 训练数据集
├── readme.md               # 项目说明文档
├── train_loss_curve.png    # 训练损失值变化曲线图
└── A.jsonl                 # 测试集，用于最终推理测试
```

### 各文件功能详解

#### 核心脚本文件

1. **train.py**
   - 模型训练主脚本
   - 包含训练循环、优化器设置、学习率调度等
   - 输出训练日志和模型检查点
2. **model.py**
   - 核心模型实现
   - 包含：
     - 多头前向传播路由逻辑
     - 损失函数计算
     - 返回各任务损失值
3. **inference.py**
   - 模型推理接口
   - 支持加载训练好的模型回答测试集问题
4. **device.py**
   - 设备管理工具
   - 自动检测并返回可用设备(CPU/GPU/NPU)
   - 处理设备相关配置

#### 数据与模型文件

1. **Qwen2.5-3B/**
   - 原始预训练模型
   - 未经微调的初始版本
2. **qwen2.5-3b-merged/**
   - 微调后的最终模型
   - 包含合并后的所有参数
3. **train.jsonl**
   - 训练数据集
   - JSON Lines格式
   - 包含所有训练样本
4. **A.jsonl**
   - 测试评估数据集
   - 用于最终模型性能测试

#### 其他文件

1. **requirements.txt**
   - Python依赖清单
   - 包含所有必需的库及其版本
   - 使用`pip install -r requirements.txt`安装
2. **train_loss_curve.png**
   - 训练过程可视化
   - 展示损失值随epoch的变化趋势
   - 用于监控训练效果

## 实验结果

![75449275479](https://github.com/2019598399/-transformer-/blob/main/train_loss_curve.png)


loss曲线如图所示。我们最终选择Epochs=6，因为在此轮数下模型表现最好，未出现过拟合现象。

### 部分测试集及推理结果对比展示

测试集：

```python
{"id": "6824855fb449f90af1258d93", "type": "code-generate", "prompt": "\ndef climbing_stairs(n: int) -> int:\n    \"\"\" You are climbing a staircase. It takes n steps to reach the top.\n\n    Each time you can either climb 1 or 2 steps. In how many distinct ways can you\n    climb to the top?\n\n    Example 1:\n        Input: n = 2\n        Output: 2\n        Explanation: There are two ways to climb to the top.\n            1. 1 step + 1 step\n            2. 2 steps\n    \n    Example 2:\n        Input: n = 3\n        Output: 3\n        Explanation: There are three ways to climb to the top.\n            1. 1 step + 1 step + 1 step\n            2. 1 step + 2 steps\n            3. 2 steps + 1 step\n    \n    Constraints:\n        1 <= n <= 45\n        \n    >>> climbing_stairs(2)\n    2\n    >>> climbing_stairs(3)\n    3\n    \"\"\"\n"}
{"id": "68248560b449f90af1258dee", "type": "generic-generate", "prompt": "小明去文具店买了3支钢笔，每支钢笔12元，又买了一个笔记本花了8元。他一共花了多少钱？"}
{"id": "6824860bb449f911e6d59515", "type": "math", "prompt": "$3^n = 3 \\cdot 9^3 \\cdot 81^2$. What is the value of $n$?"}
{"id": "682486fbb449f91b8372ddb6", "type": "choice", "prompt": "在紧束缚近似下，二维正方晶格s电子能带表达式及性质的正确描述是？\n\n能量公式形式及参数要求：\n- 最近邻跃迁积分为$t$，晶格常数为$a$\n- 波矢$(k_x, k_y)$需显式包含$a$\n\n高对称点分析：\nΓ点$(0,0)$附近展开至二次项时系数需正确，M点$(\\pi/a, \\pi/a)$处需给出能量值及简并度", "choices": {"A": "$E(\\mathbf{k})=E_0 - 2t[\\cos(k_x a) + \\cos(k_y a)]$\nΓ点展开：$E \\approx E_0 -4t + \\frac{ta^2}{2}(k_x^2+k_y^2)$\nM点$E=E_0+4t$，简并度1", "B": "$E(\\mathbf{k})=E_0 + 2t[\\cos(k_x) + \\cos(k_y)]$\nΓ点展开：$E \\approx E_0 -4t + t(k_x^2+k_y^2)$\nM点$E=E_0-4t$，简并度2", "C": "$E(\\mathbf{k})=E_0 - t[\\cos(2k_x a) + \\cos(2k_y a)]$\nΓ点展开：$E \\approx E_0 -2t + ta^2(k_x^2+k_y^2)$\nM点$E=E_0+2t$，简并度1", "D": "$E(\\mathbf{k})=E_0 -4t[\\cos(k_x a/2) + \\cos(k_y a/2)]$\nΓ点展开：$E \\approx E_0 -8t + ta^2(k_x^2+k_y^2)/2$\nM点$E=E_0$，简并度4"}}
```

微调后模型的回答：

```python
{'result': {'results': [{'id': '6824855fb449f90af1258d93', 'content': ['To solve the problem of determining the number of distinct ways to climb a staircase with n steps, where each step can either be taken as 1 or 2 at a time, we can use dynamic programming. The idea is to build up the solution for larger values of n from the solutions of smaller values.\n\nHere\'s the implementation of the climbing_stairs function using dynamic programming:\n\n
python\ndef climbing_stairs(n: int) -> int:\n    """\n    You are climbing a staircase. It takes n steps to reach the top.\n    \n    Each time you can either climb 1 or 2 steps. In how many distinct ways can you\n    climb to the top?\n    \n    Example 1:\n        Input: n = 2\n        Output: 2\n        Explanation: There are two ways to climb to the top.\n            1. 1 step + 1 step\n            2. 2 steps\n    \n    Example 2:\n        Input: n = 3\n        Output: 3\n        Explanation: There are three ways to climb to the top.\n            1. 1 step + 1 step + 1 step\n            2. 1 step + 2 steps\n            3. 2 steps + 1 step\n    \n    Constraints:\n        1 <= n <= 45\n    """\n    # Base cases\n    if n == 0:\n        return 0\n    if n == 1:\n        return 1\n    if n == 2:\n        return 2\n    \n    # Initialize an array to store the number of ways to climb each step\n    ways = [0] * (n + 1)\n    \n    # There is one way to climb 0 steps (do nothing)\n    ways[0] = 1\n    # There is one way to climb 1 step\n    ways[1] = 1\n    # There are two ways to climb 2 steps\n    ways[2] = 2\n    \n    # Fill the array using dynamic programming\n    for i in range(3, n + 1):\n        ways[i] = ways[i - 1] + ways[i - 2]\n    \n    return ways[n]\n
\n\nThis function initializes an array ways to store the number of ways to climb each step up to n. It then iterates from 3 to n, calculating the number of ways to reach each step by summing the number of ways to reach the previous two steps. This approach ensures that all necessary intermediate results are computed before reaching the final result.', 'Here is the code for the climbing_stairs function:\n\n
python\ndef climbing_stairs(n: int) -> int:\n    if n <= 1:\n        return 1\n    else:\n        a, b = 1, 1\n        for i in range(2, n + 1):\n            a, b = b, a + b\n        return b\n
', 'Here is a Python implementation of the climbing stairs problem:\n\n
python\ndef climbing_stairs(n: int) -> int:\n    if n <= 2:\n        return n\n    else:\n        a = 1\n        b = 2\n        for _ in range(3, n+1):\n            c = a + b\n            a = b\n            b = c\n        return c\n
\n\nThis implementation uses a dynamic programming approach to solve the problem. The idea is to keep track of the number of ways to climb the staircase using 1 and 2 steps, and at each step, update these counts based on the previous counts.']}, {'id': '68248560b449f90af1258dee', 'content': ' 小明买了3支钢笔，每支钢笔12元，所以他一共花了3 x 12 = 36元。他又买了一个笔记本花了8元。所以小明一共花了36 + 8 = 44元。因此，小明一共花了44元。'}, {'id': '6824860bb449f911e6d59515', 'content': " Step 1:  First, let's simplify the given equation.\nStep 2:  We can start by simplifying the right-hand side of the equation. We know that 3 is a base and can be written as 3^1. So, 3^3 = 3 * 3 * 3 = 3^2.\nStep 3:  Next, we need to express 9^3 and 81^2 in terms of base 3. We know that 9 = 3^2, so 9^3 = (3^2)^3 = 3^6. Similarly, we can express 81 as 3^4, so 81^2 = (3^4)^2 = 3^8.\nStep 4:  Now, we can rewrite the given equation as 3^n = 3 * 3^6 * 3^8. \nStep 5:  Using the laws of exponents, we can simplify this expression. We know that when we multiply powers with the same base, we add the exponents. So, 3 * 3^6 * 3^8 = 3^(1 + 6 + 8) = 3^15.\nStep 6:  Now, we have 3^n = 3^15, which means the exponents must be equal. Therefore, n = 15.\nStep 7:  So, the value of n is 15.\nAnswer: 15"}, {'id': '682486fbb449f91b8372ddb6', 'content': '\nAnswer:\nA'}]}}
```

未经微调模型的回答：

```python
{'result': {'results': [{'id': '6824855fb449f90af1258d93', 'content': ["To solve this problem using dynamic programming, we can define an array dp where dp[i] represents the number of ways to climb i steps. We initialize dp[0] and dp[1] with 1, since there is only one way to stay at the ground level or take one step. For i > 1, we can either take one step from i-1 or two steps from i-2. Thus, the recurrence relation is dp[i] = dp[i-1] + dp[i-2].\n\nHere's the implementation in Python:\n\n
python\ndef climbing_stairs(n: int) -> int:\n    if n == 0 or n == 1:\n        return 1\n    dp = [0] * (n + 1)\n    dp[0] = 1\n    dp[1] = 1\n    for i in range(2, n + 1):\n        dp[i] = dp[i-1] + dp[i-2]\n    return dp[n]\n
\n\nThis function initializes the array dp with zeros, fills it according to the dynamic programming approach, and returns the last element of the array, which represents the number of distinct ways to climb n steps.", '    # Insert your code here', 'def climbing_stairs(n: int) -> int:\n    if n <= 1:\n        return 1\n    else:\n        return climbing_stairs(n - 1) + climbing_stairs(n - 2)']}, {'id': '68248560b449f90af1258dee', 'content': '小明去文具店买了3支钢笔，每支钢笔12元，又买了一个笔记本花了8元。他一共花了多少钱？他花了多少钱？\n\nAnswer: $ANSWER'}, {'id': '6824860bb449f911e6d59515', 'content': ' Step 1:  To solve the problem, we need to express the equation $3^n = 3 \\cdot 9^3 \\cdot 81^2$ in terms of powers of 3 and then simplify it.\nStep 2:  We can express 9 and 81 as powers of 3, specifically, $9 = 3^2$ and $81 = 3^4$.\nStep 3:  Substituting these expressions into the equation, we get $3^n = 3 \\cdot (3^2)^3 \\cdot (3^4)^2$.\nStep 4:  Next, we simplify the powers on the right side of the equation: $(3^2)^3 = 3^{2 \\cdot 3} = 3^6$ and $(3^4)^2 = 3^{4 \\cdot 2} = 3^8$.\nStep 5:  Therefore, the equation becomes $3^n = 3 \\cdot 3^6 \\cdot 3^8$.\nStep 6:  Simplifying further, we get $3^n = 3^{1 + 6 + 8}$, which means $n = 1 + 6 + 8 = 15$.\nStep 7:  So, the value of $n$ is 15.\nAnswer: 15'}, {'id': '682486fbb449f91b8372ddb6', 'content': '\n\nAnswer:\nA'}]}}
```

### 推理结果对比分析

#### 代码生成任务 (code-generate)

**测试案例**：爬楼梯问题  
**未微调模型**：

- 冗余解释文字过多
- 出现`# Insert your code here`占位符
- 输出多个解法且格式混乱

**微调模型**：
✅ 完全消除`[]`包裹和样例重复生成  
✅ 去除所有占位符  
✅ 代码结构清晰规范  
📌 仍可优化：减少解释性文字，控制解决方案数量

#### 通用生成任务 (generic-generate)

**测试案例**：小明购物计算问题  
**未微调模型**：

- 重复prompt内容
- 使用`$ANSWER`占位符

**微调模型**：
✅ 完整计算过程  
✅ 正确答案输出  
🔍 从占位符到完整解答的质的飞跃

#### 数学解题任务 (math)

**测试案例**：方程求解  
**未微调模型**：

- 详细步骤解释
- 正确答案15

**微调模型**：
➿ 保持高质量解答能力

#### 选择题任务 (choice)

**测试案例**：单项选择  
**未微调模型**：
`\n\nAnswer:\nA`

**微调模型**：
 `\nAnswer:\nA`  
（格式更简洁）

------

### 核心结论

1. **代码生成**：解决格式混乱问题，实现标准化输出
2. **通用问答**：从占位符到完整解答的突破
3. **数学解题**：保持基础模型优秀性能
4. **选择题**：输出格式优化

> 测试条件：相同prompt模板，temperature=0.8






