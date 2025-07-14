# SHAP 交互图色系设置解决方案

当 SHAP 交互图的 `cmap="viridis"` 参数不生效时，可以尝试以下解决方案：

## 解决方案 1：全局设置（推荐）

```python
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import shap

# 强制设置默认色系为 viridis（全局设置）
matplotlib.pyplot.set_cmap('viridis')

# 绘制图形时不需要传入 cmap 参数
shap.summary_plot(
    shap_interaction_values_updated, 
    X_updated, 
    feature_names=updated_feature_names, 
    plot_type="dot",
    show=False  # 移除 cmap 参数
)
```

## 解决方案 2：rcParams 设置

```python
plt.rcParams.update({
    'font.size': 8, 
    'font.family': 'Times New Roman',
    'image.cmap': 'viridis'  # 设置默认色系
})
```

## 解决方案 3：绘图后手动设置

```python
# 正常绘图
shap.summary_plot(...)

# 绘图后立即设置色系
ax = plt.gca()
for collection in ax.collections:
    if hasattr(collection, 'set_cmap'):
        collection.set_cmap('viridis')
```

## 可能的原因

1. **SHAP 版本问题**：某些版本的 SHAP 对交互值不支持 `cmap` 参数
2. **matplotlib 版本兼容性**：不同版本的处理方式可能不同
3. **绘图类型特殊性**：交互值绘图可能有特殊的色系处理逻辑

## 建议使用顺序

1. 首先尝试**解决方案 1**（全局设置）
2. 如果不生效，结合**解决方案 2**（rcParams）
3. 最后尝试**解决方案 3**（手动设置）

## 验证方法

检查生成的图像是否显示了 viridis 色系的特征颜色：
- 深紫色（低值）→ 蓝色 → 绿色 → 黄色（高值）