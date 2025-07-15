import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

def manually_change_shap_colors_to_viridis():
    """
    手动将SHAP力图的颜色改为viridis色系
    """
    # viridis色系的颜色定义
    viridis_negative = "#440154"  # 深紫色（负值）
    viridis_positive = "#FDE725"  # 亮黄绿色（正值）
    
    # SHAP默认颜色
    default_positive = "#ff0051"  # 默认红色（正值）
    default_negative = "#008bfb"  # 默认蓝色（负值）
    
    # 获取当前图形
    fig = plt.gcf()
    
    # 遍历所有子图
    for ax in fig.get_axes():
        # 方法1: 修改矩形补丁 (Rectangle patches)
        for patch in ax.patches:
            if isinstance(patch, patches.Rectangle):
                current_color = patch.get_facecolor()
                # 转换为hex格式进行比较
                if hasattr(current_color, '__len__') and len(current_color) >= 3:
                    hex_color = mcolors.rgb2hex(current_color[:3])
                    
                    # 替换默认SHAP颜色为viridis颜色
                    if hex_color.lower() == default_positive.lower():
                        patch.set_facecolor(viridis_positive)
                        patch.set_edgecolor(viridis_positive)
                    elif hex_color.lower() == default_negative.lower():
                        patch.set_facecolor(viridis_negative)
                        patch.set_edgecolor(viridis_negative)
        
        # 方法2: 修改文本颜色
        for text in ax.texts:
            current_color = text.get_color()
            if hasattr(current_color, '__len__') and len(current_color) >= 3:
                hex_color = mcolors.rgb2hex(current_color[:3])
            else:
                hex_color = current_color
                
            if hex_color.lower() == default_positive.lower():
                text.set_color(viridis_positive)
            elif hex_color.lower() == default_negative.lower():
                text.set_color(viridis_negative)
        
        # 方法3: 修改线条颜色
        for line in ax.lines:
            current_color = line.get_color()
            if hasattr(current_color, '__len__') and len(current_color) >= 3:
                hex_color = mcolors.rgb2hex(current_color[:3])
            else:
                hex_color = current_color
                
            if hex_color.lower() == default_positive.lower():
                line.set_color(viridis_positive)
            elif hex_color.lower() == default_negative.lower():
                line.set_color(viridis_negative)
        
        # 方法4: 修改集合对象 (Collections)
        for collection in ax.collections:
            try:
                # 获取当前颜色
                colors = collection.get_facecolors()
                if len(colors) > 0:
                    new_colors = []
                    for color in colors:
                        hex_color = mcolors.rgb2hex(color[:3])
                        if hex_color.lower() == default_positive.lower():
                            new_colors.append(mcolors.to_rgba(viridis_positive))
                        elif hex_color.lower() == default_negative.lower():
                            new_colors.append(mcolors.to_rgba(viridis_negative))
                        else:
                            new_colors.append(color)
                    collection.set_facecolors(new_colors)
            except:
                pass
    
    # 刷新图形
    fig.canvas.draw()
    print("✓ 手动颜色修改完成：红色→黄绿色，蓝色→深紫色")

def replace_colors_in_children(obj, color_mapping):
    """
    递归地在matplotlib对象的所有子对象中替换颜色
    """
    # 如果对象有get_facecolor方法
    if hasattr(obj, 'get_facecolor') and hasattr(obj, 'set_facecolor'):
        try:
            current_color = obj.get_facecolor()
            if hasattr(current_color, '__len__') and len(current_color) >= 3:
                hex_color = mcolors.rgb2hex(current_color[:3])
                if hex_color.lower() in color_mapping:
                    obj.set_facecolor(color_mapping[hex_color.lower()])
        except:
            pass
    
    # 如果对象有get_color方法
    if hasattr(obj, 'get_color') and hasattr(obj, 'set_color'):
        try:
            current_color = obj.get_color()
            if isinstance(current_color, str):
                hex_color = current_color
            elif hasattr(current_color, '__len__') and len(current_color) >= 3:
                hex_color = mcolors.rgb2hex(current_color[:3])
            else:
                hex_color = None
                
            if hex_color and hex_color.lower() in color_mapping:
                obj.set_color(color_mapping[hex_color.lower()])
        except:
            pass
    
    # 递归处理子对象
    if hasattr(obj, 'get_children'):
        for child in obj.get_children():
            replace_colors_in_children(child, color_mapping)

def advanced_color_replacement():
    """
    高级颜色替换方法 - 递归搜索所有matplotlib对象
    """
    # 颜色映射
    color_mapping = {
        "#ff0051": "#FDE725",  # 红色 → 黄绿色
        "#008bfb": "#440154",  # 蓝色 → 深紫色
    }
    
    fig = plt.gcf()
    
    # 递归替换所有对象的颜色
    replace_colors_in_children(fig, color_mapping)
    
    # 刷新图形
    fig.canvas.draw()
    print("✓ 高级颜色替换完成")

# ==================== 主要代码 ====================

# 生成 SHAP 交互值
explainer = shap.Explainer(best_model)

# 计算 SHAP 值，保持所有特征
shap_values_explanation = explainer(X_test)  # 使用 Explainer 计算 SHAP 值

# 指定样本索引
sample_index = 1  # 示例中使用第二个样本

# 获取模型的基准值 (expected_value) 和 SHAP 值
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0]

features_to_remove = ['Distance','Unnamed: 10','Droplet size (μm)']

# 从 SHAP 值中移除指定特征
feature_names = X_test.columns
# 确保要移除的特征都存在
remove_indices = [i for i, name in enumerate(feature_names) if name in features_to_remove]
if remove_indices:
    # 过滤 SHAP 值（针对单个样本）
    shap_values_filtered_values = np.delete(shap_values_explanation.values[sample_index, :], remove_indices)
    
    # 过滤数据（针对单个样本）
    X_test_filtered_data = X_test.iloc[sample_index].drop(labels=features_to_remove)
    
    # 全局设置字体为 Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # 调整字体大小，避免标签重叠
    plt.rcParams.update({'font.size': 8})

    # 增加图形的大小，减少重叠
    plt.figure(figsize=(15, 6))

    # 绘制原始SHAP force plot（不改变参数）
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    # 手动修改颜色为viridis色系
    print("开始手动修改颜色...")
    manually_change_shap_colors_to_viridis()
    
    # 如果第一种方法不够完整，尝试高级方法
    print("尝试高级颜色替换...")
    advanced_color_replacement()
    
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot.png", dpi=600, bbox_inches='tight')
    plt.show()

    # --- HTML 版本保持原样 ---
    # HTML版本比较难手动修改，保持原来的方式
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns)
    )

    # 保存 SHAP force plot 为 HTML 文件
    shap_plot = shap.force_plot(shap_explanation_filtered[sample_index])
    
    # -----------------------------
    # !!! 重要：请在此处修改文件路径 !!!
    # -----------------------------
    save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot.html"
    shap.save_html(save_path, shap_plot)
    # -----------------------------

    # 在默认浏览器中打开 HTML 文件
    import webbrowser
    webbrowser.open(save_path)
    
else:
    print("指定要移除的特征不存在于 X_test 中。")