# SHAP力图使用viridis色系的解决方案

## 问题描述
需要将SHAP力图的默认色系（红蓝）改为viridis色系（紫-黄绿），同时尽量减少对源代码的改动。

## 解决方案汇总

### 方案1：直接使用字符串"viridis"（最简单）
```python
# matplotlib版本
shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                matplotlib=True, show=False, plot_cmap="viridis")

# HTML版本
shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], plot_cmap="viridis")
```

### 方案2：使用viridis的hex颜色值
```python
# 定义viridis色系的极值颜色
viridis_colors = ["#FDE725", "#440154"]  # 正值(黄绿), 负值(深紫)

# matplotlib版本
shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                matplotlib=True, show=False, plot_cmap=viridis_colors)

# HTML版本
shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], plot_cmap=viridis_colors)
```

### 方案3：从matplotlib获取viridis颜色
```python
import matplotlib.cm as cm
viridis_cmap = cm.get_cmap('viridis')
colors = [viridis_cmap(1.0)[:3], viridis_cmap(0.0)[:3]]  # RGB元组
colors_hex = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r,g,b in colors]

shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                matplotlib=True, show=False, plot_cmap=colors_hex)
```

### 方案4：使用其他viridis系列色系
```python
# 其他可选的色系字符串
alternative_cmaps = ["plasma", "inferno", "magma", "cividis"]

# 使用示例
shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                matplotlib=True, show=False, plot_cmap="plasma")
```

## 故障排除

### 如果颜色没有变化，请检查：

1. **SHAP版本**：确保使用较新版本的SHAP
   ```python
   import shap
   print(f"SHAP版本: {shap.__version__}")
   ```

2. **参数位置**：确保`plot_cmap`参数正确放置
   ```python
   # 正确
   shap.force_plot(..., plot_cmap="viridis")
   
   # 错误
   shap.force_plot(..., cmap="viridis")  # 参数名错误
   ```

3. **matplotlib vs HTML**：两个版本可能需要不同的处理方式

4. **错误捕获**：使用try-except来测试不同方法
   ```python
   try:
       shap.force_plot(..., plot_cmap="viridis")
   except:
       shap.force_plot(..., plot_cmap=["#FDE725", "#440154"])
   ```

## viridis色系说明

- **#440154**：深紫色（最小值/负值）
- **#31688E**：蓝绿色（中等值）
- **#35B779**：绿色（中等偏高值）
- **#FDE725**：亮黄绿色（最大值/正值）

## 推荐使用顺序

1. 先尝试 `plot_cmap="viridis"`
2. 如果不行，使用hex颜色列表 `["#FDE725", "#440154"]`
3. 最后使用其他viridis系列如 `"plasma"` 或 `"inferno"`

## 完整代码示例

最简单的修改只需要在原来的两个`shap.force_plot()`调用中添加`plot_cmap="viridis"`参数：

```python
# 原来的代码
shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                matplotlib=True, show=False)

# 修改后的代码
shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                matplotlib=True, show=False, plot_cmap="viridis")
```

这是最小的改动，符合"减少对源代码的改动"的要求。