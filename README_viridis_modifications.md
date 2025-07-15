# SHAP Force Plot Viridis Color Scheme 修改说明

## 修改概述
您的原始代码已经修改为使用 **viridis** 色系，这是一个在科学可视化中广泛使用的色彩方案，具有良好的感知一致性和色盲友好性。

## 关键修改点

### 1. 导入必要的库
```python
import matplotlib.colors as mcolors  # 添加这个导入
```

### 2. 设置 Viridis 色系
```python
# 保存原始颜色配置
original_colors = shap.plots.colors

# 创建viridis颜色配置
viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))

# 设置SHAP的颜色配置为viridis色系
shap.plots.colors.red_rgb = viridis_colors[220][:3]  # 亮黄色（正向影响）
shap.plots.colors.blue_rgb = viridis_colors[30][:3]   # 深紫色（负向影响）
```

### 3. 增强视觉效果
```python
# 增加图形大小
plt.figure(figsize=(16, 7))

# 美化图形
ax = plt.gca()
ax.set_facecolor('white')  # 白色背景
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)

# 添加标题
plt.title('SHAP Force Plot (Viridis Color Scheme)', 
         fontsize=14, fontweight='bold', pad=20)
```

### 4. 恢复原始设置
```python
# 恢复原始颜色配置
shap.plots.colors = original_colors
```

## 色彩方案说明

### Viridis 色系特点：
- **感知一致性**：颜色变化与数值变化成正比
- **色盲友好**：对色盲用户友好
- **高对比度**：在黑白打印中仍然可读
- **科学标准**：被广泛用于科学可视化

### 颜色映射：
- **深紫色** (`viridis_colors[30]`) → 负向影响（原来的蓝色）
- **亮黄色** (`viridis_colors[220]`) → 正向影响（原来的红色）

## 输出文件

修改后的代码会生成以下文件：
- `shap_force_plot_viridis.png` - 高质量 PNG 图像
- `shap_force_plot_viridis.html` - 交互式 HTML 文件

## 使用说明

1. **直接替换**：使用 `final_shap_viridis_code.py` 中的代码完全替换您的原始代码
2. **部分修改**：参考 `color_scheme_modifications.py` 了解具体需要修改的部分
3. **增强版本**：使用 `shap_force_plot_viridis.py` 获取额外的可视化功能

## 技术细节

### 颜色索引选择：
- `viridis_colors[220]`：选择viridis色谱中的亮黄色区域
- `viridis_colors[30]`：选择viridis色谱中的深紫色区域

### 图形尺寸优化：
- 从 `(15, 6)` 增加到 `(16, 7)` 以提供更好的可读性
- 字体大小从 8 增加到 10 以提高可读性

### 美观性改进：
- 白色背景提高对比度
- 移除多余的边框线条
- 添加适当的标题和间距

## 兼容性说明

这些修改与以下版本兼容：
- Python 3.6+
- matplotlib 3.0+
- shap 0.40+
- numpy 1.18+

## 故障排除

如果遇到颜色设置问题，请确保：
1. 已正确导入 `matplotlib.colors`
2. 在 `shap.force_plot()` 之前设置颜色
3. 在代码结束时恢复原始颜色配置

## 个性化定制

您可以通过修改以下参数来调整颜色：
```python
# 调整颜色索引（0-255）
shap.plots.colors.red_rgb = viridis_colors[YOUR_INDEX][:3]
shap.plots.colors.blue_rgb = viridis_colors[YOUR_INDEX][:3]
```

建议的索引范围：
- **正向影响**：180-240（黄绿到亮黄）
- **负向影响**：10-80（深紫到蓝绿）