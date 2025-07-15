# 修正版本：正确设置viridis色系的SHAP force plot

# 生成 SHAP 交互值
explainer = shap.Explainer(best_model)

# 计算 SHAP 值，保持所有特征
shap_values_explanation = explainer(X_test)

# 指定样本索引
sample_index = 1

# 获取模型的基准值 (expected_value) 和 SHAP 值
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0]

features_to_remove = ['Distance', 'Unnamed: 10', 'Droplet size (μm)']
# 从 SHAP 值中移除指定特征
feature_names = X_test.columns
remove_indices = [i for i, name in enumerate(feature_names) if name in features_to_remove]

if remove_indices:
    # 过滤 SHAP 值（针对单个样本）
    shap_values_filtered_values = np.delete(shap_values_explanation.values[sample_index, :], remove_indices)
    
    # 过滤数据（针对单个样本）
    X_test_filtered_data = X_test.iloc[sample_index].drop(labels=features_to_remove)
    
    # 全局设置字体为 Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 10})

    # === 关键修正：使用JavaScript版本的force plot来正确应用viridis色系 ===
    
    # 定义viridis色系的颜色
    viridis_positive = "#fde725"  # 亮黄色（正向影响）
    viridis_negative = "#440154"  # 深紫色（负向影响）
    
    # 使用JavaScript版本的force plot（不使用matplotlib=True）
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    plot_cmap=[viridis_positive, viridis_negative])
    
    print("已生成viridis色系的JavaScript版本force plot")
    
    # === 如果您需要保存为PNG图像，可以使用以下方法 ===
    
    # 方法1：使用matplotlib版本并手动修改颜色
    plt.figure(figsize=(16, 7))
    
    # 先绘制默认的matplotlib版本
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    # 获取当前图形对象
    ax = plt.gca()
    
    # 定义viridis颜色
    viridis_cmap = plt.cm.viridis
    pos_color = viridis_cmap(0.85)  # 亮黄色
    neg_color = viridis_cmap(0.15)  # 深紫色
    
    # 手动修改所有图形元素的颜色
    for child in ax.get_children():
        if hasattr(child, 'get_facecolor'):
            current_color = child.get_facecolor()
            if isinstance(current_color, (list, tuple, np.ndarray)) and len(current_color) >= 3:
                # 检查是否是红色系（正向影响）
                if current_color[0] > 0.7 and current_color[1] < 0.3 and current_color[2] < 0.3:
                    child.set_facecolor(pos_color)
                    if hasattr(child, 'set_edgecolor'):
                        child.set_edgecolor(pos_color)
                # 检查是否是蓝色系（负向影响）
                elif current_color[0] < 0.3 and current_color[1] < 0.3 and current_color[2] > 0.7:
                    child.set_facecolor(neg_color)
                    if hasattr(child, 'set_edgecolor'):
                        child.set_edgecolor(neg_color)
    
    # 修改文本颜色
    for text in ax.texts:
        if hasattr(text, 'get_color'):
            current_color = text.get_color()
            if isinstance(current_color, str):
                if current_color in ['red', '#ff0000', 'r']:
                    text.set_color(pos_color)
                elif current_color in ['blue', '#0000ff', 'b']:
                    text.set_color(neg_color)
            elif isinstance(current_color, (list, tuple, np.ndarray)) and len(current_color) >= 3:
                # 检查RGB值
                if current_color[0] > 0.7 and current_color[1] < 0.3 and current_color[2] < 0.3:
                    text.set_color(pos_color)
                elif current_color[0] < 0.3 and current_color[1] < 0.3 and current_color[2] > 0.7:
                    text.set_color(neg_color)
    
    # 美化图形
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    # 设置标题
    plt.title('SHAP Force Plot (Viridis Color Scheme)', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已保存viridis色系的PNG图像")

    # --- HTML 版本 ---
    # 创建新的过滤后的 Explanation 对象以供 save_html 使用
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns),
    )

    # 保存 SHAP force plot 为 HTML 文件，使用viridis颜色
    shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], 
                               plot_cmap=[viridis_positive, viridis_negative])
    
    save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.html"
    shap.save_html(save_path, shap_plot)

    # 在默认浏览器中打开 HTML 文件
    import webbrowser
    webbrowser.open(save_path)
    
else:
    print("指定的特征未在 X_test 中找到。")