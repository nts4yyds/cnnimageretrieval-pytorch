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
    
    # ======== 手动强制修改颜色 ========
    import matplotlib.colors as mcolors
    
    # 定义颜色映射
    viridis_positive = "#FDE725"  # 黄绿色（正值）
    viridis_negative = "#440154"  # 深紫色（负值）
    
    # 获取当前图形和所有轴
    fig = plt.gcf()
    
    # 方法1: 直接修改所有矩形patches的颜色
    for ax in fig.get_axes():
        for patch in ax.patches:
            # 获取当前颜色
            current_color = patch.get_facecolor()
            
            # 判断是红色还是蓝色，并替换
            if len(current_color) >= 3:
                r, g, b = current_color[0], current_color[1], current_color[2]
                
                # 检测红色系（r > 0.8, g < 0.3, b < 0.3）
                if r > 0.8 and g < 0.3 and b < 0.3:
                    patch.set_facecolor(viridis_positive)
                    patch.set_edgecolor(viridis_positive)
                
                # 检测蓝色系（r < 0.3, g < 0.6, b > 0.8）
                elif r < 0.3 and g < 0.6 and b > 0.8:
                    patch.set_facecolor(viridis_negative)
                    patch.set_edgecolor(viridis_negative)
    
    # 方法2: 修改文本颜色
    for ax in fig.get_axes():
        for text in ax.texts:
            current_color = text.get_color()
            if isinstance(current_color, tuple) and len(current_color) >= 3:
                r, g, b = current_color[0], current_color[1], current_color[2]
                
                # 红色文本改为黄绿色
                if r > 0.8 and g < 0.3 and b < 0.3:
                    text.set_color(viridis_positive)
                
                # 蓝色文本改为深紫色
                elif r < 0.3 and g < 0.6 and b > 0.8:
                    text.set_color(viridis_negative)
            elif isinstance(current_color, str):
                # 处理hex或命名颜色
                if current_color.lower() in ['red', 'r', '#ff0000', '#ff0051']:
                    text.set_color(viridis_positive)
                elif current_color.lower() in ['blue', 'b', '#0000ff', '#008bfb']:
                    text.set_color(viridis_negative)
    
    # 方法3: 暴力替换所有可能的红蓝颜色
    def force_replace_colors(obj):
        """递归替换对象中的所有红蓝颜色"""
        # 处理有颜色属性的对象
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
    
    # 对整个图形执行暴力替换
    force_replace_colors(fig)
    
    # 刷新图形
    fig.canvas.draw_idle()
    
    print("✓ 手动强制颜色修改完成！")
    print("  红色系 → 黄绿色 (#FDE725)")
    print("  蓝色系 → 深紫色 (#440154)")
    
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot.png", dpi=600, bbox_inches='tight')
    plt.show()

    # --- HTML 版本保持原样 ---
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