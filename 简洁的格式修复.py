# 简洁的格式修复解决方案

# 前面的代码保持不变
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
    
    # 定义viridis颜色
    viridis_yellow = "#fde725"  # 亮黄色 - 正向影响
    viridis_purple = "#440154"  # 深紫色 - 负向影响
    
    # 创建图形
    plt.figure(figsize=(16, 7))
    
    # 绘制原始的matplotlib版本
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    # 获取图形对象
    ax = plt.gca()
    
    # === 修改1：恢复x轴显示 ===
    
    # 确保x轴可见并设置格式
    ax.xaxis.set_visible(True)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=10)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    # 设置x轴标签
    ax.set_xlabel('Model Output Value', fontsize=12, fontweight='bold')
    
    # 添加网格线（可选）
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # === 修改2：修改颜色 ===
    
    # 修改热度条的颜色
    import matplotlib.patches as patches
    
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
    
    # 修改文本颜色，特别是"higher"和"lower"
    for text in ax.texts:
        text_content = text.get_text().lower()
        
        # 专门处理"higher"和"lower"文本
        if "higher" in text_content or "high" in text_content:
            text.set_color(viridis_yellow)  # 设置为黄色
        elif "lower" in text_content or "low" in text_content:
            text.set_color(viridis_purple)  # 设置为紫色
        else:
            # 其他文本的颜色处理
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
    
    # 修改箭头颜色
    for artist in ax.get_children():
        if hasattr(artist, 'get_facecolor'):
            facecolor = artist.get_facecolor()
            if hasattr(facecolor, '__len__') and len(facecolor) >= 3:
                # 红色箭头改为黄色
                if facecolor[0] > 0.5 and facecolor[1] < 0.5 and facecolor[2] < 0.5:
                    artist.set_facecolor(viridis_yellow)
                    if hasattr(artist, 'set_edgecolor'):
                        artist.set_edgecolor(viridis_yellow)
                # 蓝色箭头改为紫色
                elif facecolor[0] < 0.5 and facecolor[1] < 0.5 and facecolor[2] > 0.5:
                    artist.set_facecolor(viridis_purple)
                    if hasattr(artist, 'set_edgecolor'):
                        artist.set_edgecolor(viridis_purple)
    
    # === 修改3：设置边框和背景 ===
    
    # 设置边框
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置背景色
    ax.set_facecolor('white')
    
    # 设置标题
    plt.title('SHAP Force Plot (Viridis Color Scheme)', fontsize=14, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis_fixed.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已修复格式：显示了x轴坐标，并修改了higher/lower文本和箭头颜色")

    # --- HTML 版本 ---
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
    
else:
    print("指定的特征未在 X_test 中找到。")

# ===============================
# 关键修改总结：
# 1. ax.xaxis.set_visible(True) - 显示x轴
# 2. ax.set_xlabel() - 添加x轴标签
# 3. ax.grid(True, axis='x') - 添加网格线
# 4. 专门处理"higher"和"lower"文本颜色
# 5. 修改箭头颜色匹配热度条
# ===============================