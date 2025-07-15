# 最终解决方案：您只需要做以下两个关键修改

# 原始代码的问题：使用了 matplotlib=True，这使得 plot_cmap 参数无效
# 解决方案：使用JavaScript版本的force plot，或者手动修改matplotlib版本的颜色

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
    
    # === 解决方案1：使用JavaScript版本（推荐）===
    # 移除 matplotlib=True 参数，使用 plot_cmap 参数
    
    # 定义viridis色系颜色
    viridis_positive = "#fde725"  # 亮黄色（正向影响）
    viridis_negative = "#440154"  # 深紫色（负向影响）
    
    # 绘制JavaScript版本的force plot（这个会正确显示viridis颜色）
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    plot_cmap=[viridis_positive, viridis_negative])
    # 注意：这里没有 matplotlib=True 参数！
    
    print("上面显示的是viridis色系的JavaScript版本force plot")
    
    # === 解决方案2：如果您必须使用matplotlib版本保存PNG ===
    # 由于matplotlib版本无法直接应用plot_cmap，需要手动修改颜色
    
    plt.figure(figsize=(16, 7))
    
    # 先绘制默认的matplotlib版本
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    # 获取当前图形
    ax = plt.gca()
    
    # 定义viridis颜色
    viridis_pos = plt.cm.viridis(0.85)  # 亮黄色
    viridis_neg = plt.cm.viridis(0.15)  # 深紫色
    
    # 查找并修改所有的矩形（bars）
    for patch in ax.patches:
        color = patch.get_facecolor()
        # 如果是红色系（正向影响），改为viridis亮色
        if color[0] > 0.5 and color[1] < 0.5 and color[2] < 0.5:
            patch.set_facecolor(viridis_pos)
            patch.set_edgecolor(viridis_pos)
        # 如果是蓝色系（负向影响），改为viridis暗色
        elif color[0] < 0.5 and color[1] < 0.5 and color[2] > 0.5:
            patch.set_facecolor(viridis_neg)
            patch.set_edgecolor(viridis_neg)
    
    # 修改文本颜色
    for text in ax.texts:
        color = text.get_color()
        if isinstance(color, str):
            if 'red' in color or color == 'r':
                text.set_color(viridis_pos)
            elif 'blue' in color or color == 'b':
                text.set_color(viridis_neg)
        else:
            if len(color) >= 3:
                if color[0] > 0.5 and color[1] < 0.5 and color[2] < 0.5:
                    text.set_color(viridis_pos)
                elif color[0] < 0.5 and color[1] < 0.5 and color[2] > 0.5:
                    text.set_color(viridis_neg)
    
    # 美化图形
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.title('SHAP Force Plot (Viridis Color Scheme)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已保存viridis色系的PNG图像")

    # --- HTML 版本 ---
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns),
    )

    # HTML版本可以直接使用plot_cmap参数
    shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], 
                               plot_cmap=[viridis_positive, viridis_negative])
    
    save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.html"
    shap.save_html(save_path, shap_plot)

    import webbrowser
    webbrowser.open(save_path)
    
else:
    print("指定的特征未在 X_test 中找到。")

# ============================================
# 总结：
# 1. 如果您想要交互式显示viridis色系，使用JavaScript版本（移除matplotlib=True）
# 2. 如果您需要保存为PNG，使用matplotlib版本但需要手动修改颜色
# 3. HTML版本可以直接使用plot_cmap参数
# ============================================