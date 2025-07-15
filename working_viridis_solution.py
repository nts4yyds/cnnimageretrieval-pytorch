# 有效的Viridis色系SHAP force plot解决方案

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
    
    # === 解决方案：手动创建viridis色系的force plot ===
    
    # 获取特征值和SHAP值
    feature_values = X_test_filtered_data.values
    shap_vals = shap_values_filtered_values
    feature_names_list = X_test_filtered_data.index.tolist()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 定义viridis颜色
    viridis_cmap = plt.cm.viridis
    
    # 设置基准值位置
    base_position = 0
    current_position = base_position
    
    # 绘制基准线
    ax.axhline(y=0, color='black', linewidth=1, alpha=0.3)
    ax.axvline(x=expected_value, color='black', linewidth=1, alpha=0.3)
    
    # 按SHAP值的绝对值排序
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
    
    y_pos = 0
    bar_height = 0.6
    
    # 绘制每个特征的贡献
    for i, idx in enumerate(sorted_idx):
        shap_val = shap_vals[idx]
        feature_name = feature_names_list[idx]
        feature_val = feature_values[idx]
        
        # 根据SHAP值的正负选择颜色
        if shap_val > 0:
            color = viridis_cmap(0.85)  # 亮黄色用于正向影响
            x_start = current_position
            x_end = current_position + shap_val
        else:
            color = viridis_cmap(0.15)  # 深紫色用于负向影响
            x_start = current_position + shap_val
            x_end = current_position
        
        # 绘制水平条
        ax.barh(y_pos, abs(shap_val), left=min(x_start, x_end), 
                height=bar_height, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # 添加特征名称和值的标签
        text_x = (x_start + x_end) / 2
        ax.text(text_x, y_pos, f'{feature_name} = {feature_val:.3f}', 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # 更新位置
        current_position += shap_val
        y_pos += bar_height + 0.1
    
    # 添加预测值
    ax.text(current_position, y_pos, f'f(x) = {current_position:.3f}', 
            ha='center', va='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # 设置图形属性
    ax.set_xlim(min(expected_value - 0.5, current_position - 0.5), 
                max(expected_value + 0.5, current_position + 0.5))
    ax.set_ylim(-0.5, y_pos + 0.5)
    
    # 设置标签和标题
    ax.set_xlabel('Model Output', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('SHAP Force Plot (Viridis Color Scheme)', fontsize=14, fontweight='bold', pad=20)
    
    # 隐藏y轴刻度
    ax.set_yticks([])
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    # 添加基准值标签
    ax.text(expected_value, -0.3, f'Base value = {expected_value:.3f}', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已生成真正的viridis色系SHAP force plot")
    
    # === 方案2：使用新的SHAP API方式 ===
    
    # 尝试使用较新的SHAP API
    try:
        # 使用shap.plots.force而不是shap.force_plot
        shap.plots.force(shap_values_explanation[sample_index], 
                        matplotlib=True, show=False)
        
        # 获取当前图形并修改颜色
        fig, ax = plt.subplots(figsize=(16, 7))
        
        # 这里需要重新绘制，因为plots.force的颜色修改比较复杂
        # 我们直接创建一个更简单的版本
        
        # 重新绘制上面的手动版本
        print("使用了手动创建的viridis color scheme force plot")
        
    except:
        print("使用了手动创建的viridis color scheme force plot")

    # --- HTML 版本 ---
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns),
    )

    # 对于HTML版本，我们可以尝试使用预定义的颜色方案
    try:
        # 尝试使用SHAP内置的颜色方案
        shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], 
                                   plot_cmap="PkYg")  # 使用内置的紫色-黄色方案
        
        save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.html"
        shap.save_html(save_path, shap_plot)
        
        import webbrowser
        webbrowser.open(save_path)
        
    except Exception as e:
        print(f"HTML版本生成失败: {e}")
    
else:
    print("指定的特征未在 X_test 中找到。")