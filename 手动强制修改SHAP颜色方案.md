# 手动强制修改SHAP力图颜色方案

## 核心思路

**不依赖SHAP参数**，在SHAP力图生成后，通过matplotlib的底层API直接修改图形对象的颜色属性。

## 主要方法

### 方法1：RGB值检测替换（推荐）

```python
# 在shap.force_plot()之后执行
fig = plt.gcf()

# 定义viridis颜色
viridis_positive = "#FDE725"  # 黄绿色（正值）
viridis_negative = "#440154"  # 深紫色（负值）

for ax in fig.get_axes():
    for patch in ax.patches:
        current_color = patch.get_facecolor()
        
        if len(current_color) >= 3:
            r, g, b = current_color[0], current_color[1], current_color[2]
            
            # 检测红色系并替换
            if r > 0.8 and g < 0.3 and b < 0.3:
                patch.set_facecolor(viridis_positive)
                patch.set_edgecolor(viridis_positive)
            
            # 检测蓝色系并替换
            elif r < 0.3 and g < 0.6 and b > 0.8:
                patch.set_facecolor(viridis_negative)
                patch.set_edgecolor(viridis_negative)

# 刷新图形
fig.canvas.draw_idle()
```

### 方法2：递归暴力替换

```python
def force_replace_colors(obj):
    """递归替换对象中的所有红蓝颜色"""
    color_attrs = ['facecolor', 'edgecolor', 'color']
    
    for attr in color_attrs:
        if hasattr(obj, f'get_{attr}') and hasattr(obj, f'set_{attr}'):
            try:
                getter = getattr(obj, f'get_{attr}')
                setter = getattr(obj, f'set_{attr}')
                current = getter()
                
                if isinstance(current, (tuple, list)) and len(current) >= 3:
                    r, g, b = current[0], current[1], current[2]
                    if r > 0.8 and g < 0.3 and b < 0.3:  # 红色
                        setter(viridis_positive)
                    elif r < 0.3 and g < 0.6 and b > 0.8:  # 蓝色
                        setter(viridis_negative)
            except:
                pass
    
    # 递归处理子对象
    if hasattr(obj, 'get_children'):
        for child in obj.get_children():
            force_replace_colors(child)

# 使用
force_replace_colors(plt.gcf())
```

### 方法3：针对特定matplotlib对象类型

```python
import matplotlib.patches as patches

fig = plt.gcf()

for ax in fig.get_axes():
    # 处理矩形patches
    for patch in ax.patches:
        if isinstance(patch, patches.Rectangle):
            # 颜色替换逻辑
            pass
    
    # 处理文本对象
    for text in ax.texts:
        # 文本颜色替换逻辑
        pass
    
    # 处理线条对象
    for line in ax.lines:
        # 线条颜色替换逻辑
        pass
    
    # 处理集合对象
    for collection in ax.collections:
        # 集合颜色替换逻辑
        pass
```

## 完整实现代码

```python
# 在原始shap.force_plot()调用之后添加：

# ======== 手动强制修改颜色 ========
import matplotlib.colors as mcolors

# 定义viridis颜色
viridis_positive = "#FDE725"  # 黄绿色（正值）
viridis_negative = "#440154"  # 深紫色（负值）

# 获取当前图形
fig = plt.gcf()

# 方法1: 修改矩形patches
for ax in fig.get_axes():
    for patch in ax.patches:
        current_color = patch.get_facecolor()
        
        if len(current_color) >= 3:
            r, g, b = current_color[0], current_color[1], current_color[2]
            
            # 红色 → 黄绿色
            if r > 0.8 and g < 0.3 and b < 0.3:
                patch.set_facecolor(viridis_positive)
                patch.set_edgecolor(viridis_positive)
            
            # 蓝色 → 深紫色
            elif r < 0.3 and g < 0.6 and b > 0.8:
                patch.set_facecolor(viridis_negative)
                patch.set_edgecolor(viridis_negative)

# 方法2: 修改文本颜色
for ax in fig.get_axes():
    for text in ax.texts:
        current_color = text.get_color()
        if isinstance(current_color, tuple) and len(current_color) >= 3:
            r, g, b = current_color[0], current_color[1], current_color[2]
            
            if r > 0.8 and g < 0.3 and b < 0.3:
                text.set_color(viridis_positive)
            elif r < 0.3 and g < 0.6 and b > 0.8:
                text.set_color(viridis_negative)

# 刷新图形
fig.canvas.draw_idle()

print("✓ 手动强制颜色修改完成！")
```

## 优势

1. **完全独立**：不依赖SHAP版本或参数支持
2. **通用性强**：适用于所有matplotlib生成的SHAP图
3. **精确控制**：可以精确指定任何颜色
4. **兼容性好**：不会破坏原有代码逻辑

## 注意事项

### 颜色检测阈值

RGB检测阈值可能需要根据实际情况调整：

```python
# 红色检测（可调整）
if r > 0.8 and g < 0.3 and b < 0.3:

# 蓝色检测（可调整）  
if r < 0.3 and g < 0.6 and b > 0.8:
```

### 刷新图形

修改后必须刷新图形：

```python
fig.canvas.draw_idle()  # 或者
fig.canvas.draw()
```

### 文本对象处理

文本颜色可能是字符串格式，需要特殊处理：

```python
if isinstance(current_color, str):
    if current_color.lower() in ['red', '#ff0000', '#ff0051']:
        text.set_color(viridis_positive)
```

## viridis色系参考

- **#440154** - 深紫色（最小值）
- **#482878** - 紫色
- **#3E4A89** - 深蓝紫
- **#31688E** - 蓝绿色
- **#26828E** - 青色
- **#1F9E89** - 青绿色
- **#35B779** - 绿色
- **#6DCD59** - 黄绿色
- **#B4DE2C** - 亮黄绿
- **#FDE725** - 亮黄色（最大值）

## 使用建议

1. **先执行SHAP绘图**：完全按原来的方式调用 `shap.force_plot()`
2. **再执行颜色替换**：在绘图完成后执行手动颜色修改
3. **最后保存和显示**：执行 `plt.savefig()` 和 `plt.show()`

这种方法确保了对原代码的最小干扰，同时实现了强制的颜色替换效果。