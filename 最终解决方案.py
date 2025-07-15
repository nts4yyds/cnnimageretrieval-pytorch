# 最终解决方案：多种方法确保viridis色系生效

# 您的原始代码前面部分保持不变
explainer = shap.Explainer(best_model)
shap_values_explanation = explainer(X_test)
sample_index = 1

expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0]

features_to_remove = ['Distance', 'Unnamed: 10', 'Droplet size (μm)']
feature_names = X_test.columns
remove_indices = [i for i, name in enumerate(feature_names) if name in features_to_remove]

if remove_indices:
    shap_values_filtered_values = np.delete(shap_values_explanation.values[sample_index, :], remove_indices)
    X_test_filtered_data = X_test.iloc[sample_index].drop(labels=features_to_remove)
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 10})

    # ====== 关键修改1：尝试内置颜色方案 ======
    
    # 最简单的方法：使用SHAP内置的类viridis颜色方案
    print("方案1：使用PkYg颜色方案（紫色-黄色）")
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    plot_cmap="PkYg")
    
    print("\n方案2：使用YlDp颜色方案（黄色-深紫色）")
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    plot_cmap="YlDp")
    
    # ====== 关键修改2：如果需要PNG图像，使用matplotlib版本 ======
    
    # 图形大小调整
    plt.figure(figsize=(16, 7))
    
    # 先绘制原始图形
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    # 获取图形对象
    ax = plt.gca()
    
    # 定义viridis颜色
    viridis_yellow = "#fde725"  # 亮黄色 - 正向影响
    viridis_purple = "#440154"  # 深紫色 - 负向影响
    
    # 修改所有图形元素的颜色
    import matplotlib.patches as patches
    
    # 修改矩形元素
    for child in ax.get_children():
        if isinstance(child, patches.Rectangle):
            facecolor = child.get_facecolor()
            if len(facecolor) >= 3:
                # 红色系（正向）→ 黄色
                if facecolor[0] > 0.5 and facecolor[1] < 0.5 and facecolor[2] < 0.5:
                    child.set_facecolor(viridis_yellow)
                    child.set_edgecolor(viridis_yellow)
                # 蓝色系（负向）→ 紫色
                elif facecolor[0] < 0.5 and facecolor[1] < 0.5 and facecolor[2] > 0.5:
                    child.set_facecolor(viridis_purple)
                    child.set_edgecolor(viridis_purple)
    
    # 修改文本颜色
    for text in ax.texts:
        color = text.get_color()
        if isinstance(color, str):
            if color in ['red', 'r', '#ff0000']:
                text.set_color(viridis_yellow)
            elif color in ['blue', 'b', '#0000ff']:
                text.set_color(viridis_purple)
        elif hasattr(color, '__len__') and len(color) >= 3:
            if color[0] > 0.5 and color[1] < 0.5 and color[2] < 0.5:
                text.set_color(viridis_yellow)
            elif color[0] < 0.5 and color[1] < 0.5 and color[2] > 0.5:
                text.set_color(viridis_purple)
    
    # 美化图形
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.title('SHAP Force Plot (Viridis Color Scheme)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()

    # ====== 关键修改3：备选方案 - 创建类似的可视化 ======
    
    print("\n方案3：自定义viridis色系的SHAP可视化")
    
    # 如果上面的都不行，我们手动创建一个类似的可视化
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 获取数据
    features = X_test_filtered_data.index.tolist()
    shap_vals = shap_values_filtered_values
    feature_vals = X_test_filtered_data.values
    
    # 按SHAP值的绝对值排序
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
    
    # 计算累积位置
    cumulative = expected_value
    y_positions = []
    
    for i, idx in enumerate(sorted_idx):
        shap_val = shap_vals[idx]
        feature_name = features[idx]
        feature_val = feature_vals[idx]
        
        # 选择颜色
        if shap_val > 0:
            color = viridis_yellow
        else:
            color = viridis_purple
        
        # 绘制条形
        width = abs(shap_val)
        left = cumulative if shap_val > 0 else cumulative + shap_val
        
        bar = ax.barh(i, width, left=left, height=0.8, 
                     color=color, alpha=0.8, edgecolor='white', linewidth=1)
        
        # 添加标签
        text_x = cumulative + shap_val/2
        ax.text(text_x, i, f'{feature_name} = {feature_val:.3f}', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        cumulative += shap_val
        y_positions.append(feature_name)
    
    # 添加基准值和预测值标记
    ax.axvline(x=expected_value, color='black', linestyle='--', alpha=0.5, 
               label=f'Base value: {expected_value:.3f}')
    ax.axvline(x=cumulative, color='red', linestyle='--', alpha=0.5, 
               label=f'Prediction: {cumulative:.3f}')
    
    # 设置标签
    ax.set_yticks(range(len(y_positions)))
    ax.set_yticklabels(y_positions)
    ax.set_xlabel('Model Output', fontsize=12, fontweight='bold')
    ax.set_title('SHAP Force Plot - Viridis Color Scheme', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_custom_viridis.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()

    # ====== HTML版本 ======
    
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns),
    )

    # HTML版本使用PkYg颜色方案
    shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], 
                               plot_cmap="PkYg")
    
    save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.html"
    shap.save_html(save_path, shap_plot)

    import webbrowser
    webbrowser.open(save_path)
    
    print("已生成多种viridis色系的可视化方案")
    
else:
    print("指定的特征未在 X_test 中找到。")

# =====================================================
# 总结：
# 1. 方案1和2：使用内置颜色方案（PkYg, YlDp）
# 2. 方案3：手动修改matplotlib版本的颜色
# 3. 方案4：自定义创建类似的可视化
# 这些方案中至少有一种应该能够显示viridis色系！
# =====================================================