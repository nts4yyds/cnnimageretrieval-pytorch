# 在您的原始代码中，在 shap.force_plot() 调用之前添加以下代码：

# === 添加的导入 ===
import matplotlib.colors as mcolors

# === 在绘制 SHAP force plot 之前添加这段代码 ===
# 保存原始颜色配置
original_colors = shap.plots.colors

# 创建viridis颜色配置
viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))

# 设置SHAP的颜色配置为viridis色系
shap.plots.colors.red_rgb = viridis_colors[220][:3]  # 使用viridis的亮黄色替代红色（正向影响）
shap.plots.colors.blue_rgb = viridis_colors[30][:3]   # 使用viridis的深紫色替代蓝色（负向影响）

# 您的原始 shap.force_plot() 代码保持不变
# 绘制 SHAP force plot，移除 'Distance' 列后的特征，调整颜色映射
shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                matplotlib=True, show=False)

# === 在 plt.show() 之前添加这些美化代码 ===
# 美化图形
ax = plt.gca()
ax.set_facecolor('white')  # 设置背景为白色

# 调整图形边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)

# 设置标题
plt.title('SHAP Force Plot (Viridis Color Scheme)', 
         fontsize=14, fontweight='bold', pad=20)

# 调整布局
plt.tight_layout()

# 您的原始保存和显示代码
plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# === 在代码最后添加恢复原始颜色配置 ===
# 恢复原始颜色配置
shap.plots.colors = original_colors

# ===============================
# 完整的修改说明：
# 1. 添加 matplotlib.colors 导入
# 2. 在 shap.force_plot() 之前设置 viridis 颜色
# 3. 在 plt.show() 之前添加美化代码
# 4. 在最后恢复原始颜色配置
# 5. 更改保存文件名为 shap_force_plot_viridis.png
# ===============================