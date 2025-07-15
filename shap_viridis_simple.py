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

    # 绘制 SHAP force plot，直接使用 "viridis" 字符串
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False, plot_cmap="viridis")
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

    # 保存 SHAP force plot 为 HTML 文件，使用viridis色系
    shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], plot_cmap="viridis")
    
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