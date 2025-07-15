# 简单直接的解决方案：确保viridis色系生效

# 生成 SHAP 交互值
explainer = shap.Explainer(best_model)
shap_values_explanation = explainer(X_test)
sample_index = 1

# 获取模型的基准值
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0]

features_to_remove = ['Distance', 'Unnamed: 10', 'Droplet size (μm)']
feature_names = X_test.columns
remove_indices = [i for i, name in enumerate(feature_names) if name in features_to_remove]

if remove_indices:
    # 过滤 SHAP 值和数据
    shap_values_filtered_values = np.delete(shap_values_explanation.values[sample_index, :], remove_indices)
    X_test_filtered_data = X_test.iloc[sample_index].drop(labels=features_to_remove)
    
    # 设置字体
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 10})
    
    # === 方案1：尝试多种内置颜色方案 ===
    
    # 首先尝试JavaScript版本的不同颜色方案
    print("尝试不同的颜色方案：")
    
    # 方案1a：使用PkYg（紫色-黄色-绿色）
    print("1. 使用PkYg颜色方案：")
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    plot_cmap="PkYg")
    
    # 方案1b：使用YlDp（黄色-深紫色）
    print("2. 使用YlDp颜色方案：")
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    plot_cmap="YlDp")
    
    # 方案1c：使用自定义viridis颜色
    print("3. 使用自定义viridis颜色：")
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    plot_cmap=["#440154", "#fde725"])  # 深紫色到亮黄色
    
    # === 方案2：matplotlib版本的有效修改 ===
    
    plt.figure(figsize=(16, 7))
    
    # 绘制matplotlib版本
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    # 获取当前图形的所有子元素
    fig = plt.gcf()
    ax = plt.gca()
    
    # 定义viridis颜色
    viridis_yellow = "#fde725"  # 亮黄色
    viridis_purple = "#440154"  # 深紫色
    
    # 方法1：通过修改所有路径对象
    import matplotlib.patches as patches
    
    for child in ax.get_children():
        if isinstance(child, patches.Rectangle):
            facecolor = child.get_facecolor()
            # 检查是否是红色系（正向影响）
            if len(facecolor) >= 3 and facecolor[0] > 0.5 and facecolor[1] < 0.5 and facecolor[2] < 0.5:
                child.set_facecolor(viridis_yellow)
                child.set_edgecolor(viridis_yellow)
            # 检查是否是蓝色系（负向影响）
            elif len(facecolor) >= 3 and facecolor[0] < 0.5 and facecolor[1] < 0.5 and facecolor[2] > 0.5:
                child.set_facecolor(viridis_purple)
                child.set_edgecolor(viridis_purple)
    
    # 方法2：通过修改所有多边形对象
    for collection in ax.collections:
        colors = collection.get_facecolors()
        if len(colors) > 0:
            new_colors = []
            for color in colors:
                if len(color) >= 3:
                    # 红色系改为黄色
                    if color[0] > 0.5 and color[1] < 0.5 and color[2] < 0.5:
                        new_colors.append(plt.matplotlib.colors.to_rgba(viridis_yellow))
                    # 蓝色系改为紫色
                    elif color[0] < 0.5 and color[1] < 0.5 and color[2] > 0.5:
                        new_colors.append(plt.matplotlib.colors.to_rgba(viridis_purple))
                    else:
                        new_colors.append(color)
                else:
                    new_colors.append(color)
            collection.set_facecolors(new_colors)
    
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
    
    # 设置标题
    plt.title('SHAP Force Plot (Viridis Color Scheme)', fontsize=14, fontweight='bold', pad=20)
    
    # 美化图形
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已尝试修改matplotlib版本的颜色")
    
    # === 方案3：如果上面都不行，使用waterfall plot替代 ===
    
    print("4. 尝试使用waterfall plot作为替代方案：")
    
    # 创建waterfall plot，更容易控制颜色
    plt.figure(figsize=(16, 8))
    
    # 获取特征名称和SHAP值
    feature_names_list = X_test_filtered_data.index.tolist()
    shap_vals = shap_values_filtered_values
    
    # 按SHAP值排序
    sorted_idx = np.argsort(np.abs(shap_vals))
    
    # 绘制waterfall图
    cumulative = expected_value
    positions = []
    values = []
    colors = []
    labels = []
    
    for idx in sorted_idx:
        shap_val = shap_vals[idx]
        feature_name = feature_names_list[idx]
        
        positions.append(cumulative)
        values.append(shap_val)
        
        # 使用viridis颜色
        if shap_val > 0:
            colors.append(viridis_yellow)
        else:
            colors.append(viridis_purple)
        
        labels.append(f'{feature_name}\n({shap_val:.3f})')
        cumulative += shap_val
    
    # 绘制条形图
    bars = plt.bar(range(len(values)), values, bottom=positions, 
                   color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # 添加标签
    for i, (bar, label) in enumerate(zip(bars, labels)):
        height = bar.get_height()
        y_pos = bar.get_y() + height/2
        plt.text(i, y_pos, label, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
    
    # 添加基准线和最终值
    plt.axhline(y=expected_value, color='black', linestyle='--', alpha=0.5, label=f'Base value: {expected_value:.3f}')
    plt.axhline(y=cumulative, color='red', linestyle='--', alpha=0.5, label=f'Prediction: {cumulative:.3f}')
    
    plt.title('SHAP Waterfall Plot (Viridis Color Scheme)', fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('SHAP Value', fontsize=12)
    plt.xticks(range(len(labels)), [label.split('\n')[0] for label in labels], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_waterfall_viridis.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已生成viridis色系的waterfall plot作为替代方案")

    # --- HTML 版本 ---
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns),
    )

    # HTML版本尝试不同的颜色方案
    try:
        # 使用PkYg颜色方案
        shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], 
                                   plot_cmap="PkYg")
        
        save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.html"
        shap.save_html(save_path, shap_plot)
        
        import webbrowser
        webbrowser.open(save_path)
        
        print("HTML版本已生成，使用了PkYg颜色方案")
        
    except Exception as e:
        print(f"HTML版本生成失败: {e}")
    
else:
    print("指定的特征未在 X_test 中找到。")