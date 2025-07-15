# 最简单的修改方案 - 只需在您的原代码中添加这些行

# 您的原始代码...
# （保持所有原始代码不变，直到 plt.show() 之前）

# 在 plt.show() 之前添加以下代码：

# === 添加这些行来修复格式 ===

# 获取当前图形对象
ax = plt.gca()

# 修改1：显示x轴坐标
ax.xaxis.set_visible(True)
ax.tick_params(axis='x', labelsize=10)
ax.set_xlabel('Model Output Value', fontsize=12, fontweight='bold')

# 修改2：修改"higher"和"lower"文本颜色
viridis_yellow = "#fde725"  # 亮黄色
viridis_purple = "#440154"  # 深紫色

for text in ax.texts:
    text_content = text.get_text().lower()
    if "higher" in text_content or "high" in text_content:
        text.set_color(viridis_yellow)
    elif "lower" in text_content or "low" in text_content:
        text.set_color(viridis_purple)

# 修改3：修改箭头颜色
for artist in ax.get_children():
    if hasattr(artist, 'get_facecolor'):
        facecolor = artist.get_facecolor()
        if hasattr(facecolor, '__len__') and len(facecolor) >= 3:
            # 红色箭头改为黄色
            if facecolor[0] > 0.5 and facecolor[1] < 0.5 and facecolor[2] < 0.5:
                artist.set_facecolor(viridis_yellow)
            # 蓝色箭头改为紫色
            elif facecolor[0] < 0.5 and facecolor[1] < 0.5 and facecolor[2] > 0.5:
                artist.set_facecolor(viridis_purple)

# 修改4：设置边框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# 然后继续您的原始代码：plt.show() 等等...

# ========================================
# 完整示例：在您的原始代码中的应用位置
# ========================================

# 生成 SHAP 交互值
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
    
    plt.figure(figsize=(16, 7))

    # 绘制 SHAP force plot
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    # ============ 在这里添加格式修复代码 ============
    
    ax = plt.gca()
    
    # 显示x轴坐标
    ax.xaxis.set_visible(True)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlabel('Model Output Value', fontsize=12, fontweight='bold')
    
    # 定义viridis颜色
    viridis_yellow = "#fde725"
    viridis_purple = "#440154"
    
    # 修改"higher"和"lower"文本颜色
    for text in ax.texts:
        text_content = text.get_text().lower()
        if "higher" in text_content or "high" in text_content:
            text.set_color(viridis_yellow)
        elif "lower" in text_content or "low" in text_content:
            text.set_color(viridis_purple)
    
    # 修改箭头颜色
    for artist in ax.get_children():
        if hasattr(artist, 'get_facecolor'):
            facecolor = artist.get_facecolor()
            if hasattr(facecolor, '__len__') and len(facecolor) >= 3:
                if facecolor[0] > 0.5 and facecolor[1] < 0.5 and facecolor[2] < 0.5:
                    artist.set_facecolor(viridis_yellow)
                elif facecolor[0] < 0.5 and facecolor[1] < 0.5 and facecolor[2] > 0.5:
                    artist.set_facecolor(viridis_purple)
    
    # 设置边框
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # ============ 格式修复代码结束 ============
    
    # 继续您的原始代码
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_fixed.png", 
               dpi=600, bbox_inches='tight')
    plt.show()
    
    # 其余HTML代码保持不变...
    
else:
    print("指定的特征未在 X_test 中找到。")