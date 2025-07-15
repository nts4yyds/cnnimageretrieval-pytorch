# 生成 SHAP 交互值
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import shap
import webbrowser

explainer = shap.Explainer(best_model)

# 计算 SHAP 值，保持所有特征
shap_values_explanation = explainer(X_test)  # 使用 Explainer 计算 SHAP 值

# 指定样本索引
sample_index = 1  # 示例中使用第二个样本

# 获取模型的基准值 (expected_value) 和 SHAP 值
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0]

features_to_remove = ['Distance', 'Unnamed: 10', 'Droplet size (μm)']
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
    plt.rcParams.update({'font.size': 10})  # 稍微增加字体大小以提高可读性
    
    # 设置 viridis 色系的自定义颜色映射
    # 创建自定义颜色映射，用于正负值的区分
    colors_positive = plt.cm.viridis(np.linspace(0.6, 1.0, 128))  # 高亮的viridis色调用于正值
    colors_negative = plt.cm.viridis(np.linspace(0.0, 0.4, 128))  # 暗的viridis色调用于负值
    colors_combined = np.vstack((colors_negative, colors_positive))
    viridis_custom = mcolors.LinearSegmentedColormap.from_list('viridis_custom', colors_combined)
    
    # 设置matplotlib的默认颜色映射
    plt.rcParams['image.cmap'] = 'viridis'
    
    # 增加图形的大小，优化布局
    plt.figure(figsize=(16, 7))
    
    # 手动设置SHAP的颜色配置
    # 这里我们需要直接修改SHAP的颜色设置
    original_colors = shap.plots.colors
    
    # 创建基于viridis的颜色配置
    viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))
    
    # 设置自定义颜色配置
    shap.plots.colors.red_rgb = viridis_colors[200][:3]  # 用viridis的亮色替代红色
    shap.plots.colors.blue_rgb = viridis_colors[50][:3]   # 用viridis的暗色替代蓝色
    
    # 绘制 SHAP force plot，使用viridis色系
    try:
        shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                        matplotlib=True, show=False)
        
        # 进一步美化图形
        ax = plt.gca()
        ax.set_facecolor('white')  # 设置背景为白色
        
        # 调整图形边框和网格
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        
        # 设置标题
        plt.title('SHAP Force Plot (Viridis Color Scheme)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存高质量图像
        plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.png", 
                   dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()
        
    except Exception as e:
        print(f"Error creating matplotlib plot: {e}")
        # 如果matplotlib版本出现问题，使用默认绘图
        shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                        matplotlib=True, show=False)
        plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.png", 
                   dpi=600, bbox_inches='tight')
        plt.show()
    
    finally:
        # 恢复原始颜色配置
        shap.plots.colors = original_colors

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

    # 保存 SHAP force plot 为 HTML 文件（HTML版本会自动应用主题）
    shap_plot = shap.force_plot(shap_explanation_filtered[sample_index])
    
    # 自定义HTML输出的CSS样式以应用viridis色系
    html_content = f"""
    <style>
    .force-plot {{
        font-family: 'Times New Roman', serif;
        background-color: white;
    }}
    .force-plot .positive {{
        fill: #440154;  /* viridis dark purple */
    }}
    .force-plot .negative {{
        fill: #21918c;  /* viridis teal */
    }}
    .force-plot .neutral {{
        fill: #5dc863;  /* viridis green */
    }}
    </style>
    """
    
    # 保存 HTML 文件
    save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.html"
    shap.save_html(save_path, shap_plot)
    
    # 在默认浏览器中打开 HTML 文件
    webbrowser.open(save_path)
    
else:
    print("指定的特征未在 X_test 中找到。")

# 额外的可视化：创建一个带有viridis色系的summary plot
def create_viridis_summary_plot():
    """创建一个使用viridis色系的SHAP summary plot"""
    plt.figure(figsize=(12, 8))
    
    # 设置viridis色系
    plt.rcParams['image.cmap'] = 'viridis'
    
    # 过滤数据用于summary plot
    shap_values_filtered_summary = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_summary = X_test.drop(columns=features_to_remove)
    
    # 创建过滤后的explanation对象
    shap_explanation_summary = shap.Explanation(
        values=shap_values_filtered_summary,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_summary,
        feature_names=list(data_filtered_summary.columns),
    )
    
    # 绘制summary plot
    shap.summary_plot(shap_explanation_summary, data_filtered_summary, 
                     cmap='viridis', show=False)
    
    plt.title('SHAP Summary Plot (Viridis Color Scheme)', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_summary_plot_viridis.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()

# 调用函数创建summary plot
create_viridis_summary_plot()