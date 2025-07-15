# 生成 SHAP 交互值
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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
    plt.rcParams.update({'font.size': 10})
    
    # 增加图形的大小，减少重叠
    plt.figure(figsize=(16, 7))

    # 方法1：使用非matplotlib版本的force plot（推荐）
    # 这种方法可以直接使用plot_cmap参数
    try:
        # 使用viridis颜色的十六进制值
        viridis_positive = "#fde725"  # 亮黄色（viridis最亮）
        viridis_negative = "#440154"  # 深紫色（viridis最暗）
        
        # 绘制非matplotlib版本的force plot
        shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                        plot_cmap=[viridis_positive, viridis_negative], 
                        matplotlib=False)
        
        print("使用了JavaScript版本的force plot with viridis colors")
        
    except Exception as e:
        print(f"JavaScript版本失败: {e}")
        
        # 方法2：如果必须使用matplotlib版本，则需要手动修改颜色
        # 先绘制默认的matplotlib force plot
        shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                        matplotlib=True, show=False)
        
        # 获取当前的axes对象
        ax = plt.gca()
        
        # 定义viridis颜色
        viridis_cmap = plt.cm.viridis
        viridis_colors = viridis_cmap(np.linspace(0, 1, 256))
        positive_color = viridis_colors[220][:3]  # 亮黄色
        negative_color = viridis_colors[30][:3]   # 深紫色
        
        # 修改图形中的颜色
        import matplotlib.patches as mpatches
        
        # 遍历所有的图形元素
        for child in ax.get_children():
            if isinstance(child, mpatches.Rectangle):
                # 获取当前颜色
                current_color = child.get_facecolor()
                
                # 判断是否是正向影响的颜色（通常是红色系）
                if current_color[0] > 0.5 and current_color[1] < 0.5:  # 红色系
                    child.set_facecolor(positive_color)
                    child.set_edgecolor(positive_color)
                # 判断是否是负向影响的颜色（通常是蓝色系）
                elif current_color[0] < 0.5 and current_color[2] > 0.5:  # 蓝色系
                    child.set_facecolor(negative_color)
                    child.set_edgecolor(negative_color)
        
        # 修改文本颜色
        for text in ax.texts:
            current_color = text.get_color()
            if isinstance(current_color, str):
                if current_color.lower() in ['red', '#ff0000'] or 'red' in current_color.lower():
                    text.set_color(positive_color)
                elif current_color.lower() in ['blue', '#0000ff'] or 'blue' in current_color.lower():
                    text.set_color(negative_color)
        
        # 美化图形
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        
        # 设置标题
        plt.title('SHAP Force Plot (Viridis Color Scheme)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.png", 
                   dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("使用了matplotlib版本的force plot with modified colors")

    # --- HTML 版本（使用viridis颜色）---
    # 创建新的过滤后的 Explanation 对象以供 save_html 使用
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns),
    )

    # 保存 SHAP force plot 为 HTML 文件，使用viridis颜色
    viridis_positive = "#fde725"  # 亮黄色
    viridis_negative = "#440154"  # 深紫色
    
    shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], 
                               plot_cmap=[viridis_positive, viridis_negative])
    
    # 保存路径
    save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.html"
    shap.save_html(save_path, shap_plot)

    # 在默认浏览器中打开 HTML 文件
    import webbrowser
    webbrowser.open(save_path)
    
else:
    print("指定的特征未在 X_test 中找到。")