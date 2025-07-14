import matplotlib.pyplot as plt
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

# 方法1：设置全局色系并强制应用 viridis
plt.rcParams.update({
    'font.size': 8, 
    'font.family': 'Times New Roman',
    'image.cmap': 'viridis'  # 设置默认色系为 viridis
})

# 方法2：手动设置 matplotlib 色系
import matplotlib.cm as cm
plt.cm.get_cmap('viridis')

plt.figure(figsize=(18, 8))

# 方法3：对于交互值，尝试使用不同的绘图方式
try:
    # 首先尝试直接设置 cmap
    shap.summary_plot(
        shap_interaction_values_updated, 
        X_updated, 
        feature_names=updated_feature_names, 
        plot_type="dot",
        cmap="viridis",
        show=False
    )
except:
    # 如果上述方法不工作，使用手动设置
    import matplotlib
    matplotlib.pyplot.set_cmap('viridis')
    
    shap.summary_plot(
        shap_interaction_values_updated, 
        X_updated, 
        feature_names=updated_feature_names, 
        plot_type="dot",
        show=False
    )

# 方法4：手动修改当前图形的色系
ax = plt.gca()
# 获取当前的散点图集合并设置色系
for collection in ax.collections:
    collection.set_cmap('viridis')

# 强制设置 Y 轴刻度标签的字体大小
plt.gca().yaxis.set_tick_params(labelsize=8)

# 调整布局
plt.tight_layout(pad=3.0)
plt.subplots_adjust(left=0.2, top=0.85)

plt.savefig('shap_interaction_plot_viridis.png', dpi=600, bbox_inches='tight')
plt.show()