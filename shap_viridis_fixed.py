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

# 方法1: 使用viridis色系的具体hex颜色（尝试不同的顺序）
viridis_colors_v1 = ["#440154", "#FDE725"]  # 负值(紫), 正值(黄绿)
viridis_colors_v2 = ["#FDE725", "#440154"]  # 正值(黄绿), 负值(紫)

# 方法2: 使用更多viridis中间色
viridis_colors_v3 = ["#482878", "#3E4A89", "#31688E", "#26828E", "#1F9E89", "#35B779", "#6DCD59", "#B4DE2C", "#FDE725"]

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

    # 方法1: 直接使用viridis色系（推荐）
    try:
        shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                        matplotlib=True, show=False, plot_cmap="viridis")
        print("使用方法1：直接指定'viridis'")
    except:
        # 方法2: 使用hex颜色列表
        try:
            shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                            matplotlib=True, show=False, plot_cmap=viridis_colors_v2)
            print("使用方法2：hex颜色列表")
        except:
            # 方法3: 默认方式（如果以上都失败）
            shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                            matplotlib=True, show=False)
            print("使用方法3：默认颜色")
    
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot.png", dpi=600, bbox_inches='tight')
    plt.show()

    # --- HTML 版本 ---
    # 创建新的过滤后的 Explanation 对象以供 save_html 使用
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns)
    )

    # HTML版本也尝试不同方法
    try:
        shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], plot_cmap="viridis")
    except:
        try:
            shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], plot_cmap=viridis_colors_v2)
        except:
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

# 额外方法：如果上述方法都不行，可以尝试使用matplotlib设置全局colormap
plt.rcParams['image.cmap'] = 'viridis'