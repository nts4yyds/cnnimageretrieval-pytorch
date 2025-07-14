# 计算交互 SHAP 值
shap_interaction_values = explainer.shap_interaction_values(X)

# 将特征名称转为 numpy 数组
feature_names_np = np.array(X.columns)

# 去除第8列(Distance)相关的交互值和特征
# (假设 'Distance' 是第8个特征，索引为7)
shap_interaction_values_updated = shap_interaction_values[:, :6, :6]
X_updated = X.iloc[:, :6]
updated_feature_names = feature_names_np[:6]

# 设置全局字体为 Times New Roman，调整字体大小为 8
plt.rcParams.update({'font.size': 8, 'font.family': 'Times New Roman'})

# 绘制 SHAP summary plot，使用 viridis 色系
plt.figure(figsize=(18, 8))
shap.summary_plot(
    shap_interaction_values_updated, 
    X_updated, 
    feature_names=updated_feature_names, 
    plot_type="dot",
    cmap="viridis",  # 设置色系为 viridis
    show=False
)

# 强制设置 Y 轴刻度标签的字体大小与其他标题一致
plt.gca().yaxis.set_tick_params(labelsize=8)

# 调整布局，确保顶部和左侧的标题有足够的空间显示
plt.tight_layout(pad=3.0)

# 进一步调整顶部和左侧的空间，确保标题完整显示
plt.subplots_adjust(left=0.2, top=0.85)

plt.savefig('shap_interaction_plot.png', dpi=600, bbox_inches='tight')
# 显示图表
plt.show()