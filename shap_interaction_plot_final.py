import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import shap

# 计算交互 SHAP 值
shap_interaction_values = explainer.shap_interaction_values(X)

# 将特征名称转为 numpy 数组
feature_names_np = np.array(X.columns)

# 去除第8列(Distance)相关的交互值和特征
shap_interaction_values_updated = shap_interaction_values[:, :6, :6]
X_updated = X.iloc[:, :6]
updated_feature_names = feature_names_np[:6]

# 强制设置默认色系为 viridis（全局设置）
matplotlib.pyplot.set_cmap('viridis')

# 设置全局字体
plt.rcParams.update({
    'font.size': 8, 
    'font.family': 'Times New Roman'
})

plt.figure(figsize=(18, 8))

# 绘制 SHAP summary plot（不传入cmap参数，使用全局设置）
shap.summary_plot(
    shap_interaction_values_updated, 
    X_updated, 
    feature_names=updated_feature_names, 
    plot_type="dot",
    show=False
)

# 强制设置 Y 轴刻度标签的字体大小
plt.gca().yaxis.set_tick_params(labelsize=8)

# 调整布局
plt.tight_layout(pad=3.0)
plt.subplots_adjust(left=0.2, top=0.85)

plt.savefig('shap_interaction_plot_viridis.png', dpi=600, bbox_inches='tight')
plt.show()