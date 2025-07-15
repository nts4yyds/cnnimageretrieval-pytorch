# 完善格式的SHAP Force Plot解决方案

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
    
    # 定义viridis颜色
    viridis_yellow = "#fde725"  # 亮黄色 - 正向影响
    viridis_purple = "#440154"  # 深紫色 - 负向影响
    
    # 创建图形
    plt.figure(figsize=(16, 7))
    
    # 绘制原始的matplotlib版本
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    # 获取图形对象
    ax = plt.gca()
    
    # 修改所有图形元素的颜色
    import matplotlib.patches as patches
    
    # 修改矩形元素（热度条）
    for child in ax.get_children():
        if isinstance(child, patches.Rectangle):
            facecolor = child.get_facecolor()
            if len(facecolor) >= 3:
                # 红色系（正向）→ 黄色
                if facecolor[0] > 0.5 and facecolor[1] < 0.5 and facecolor[2] < 0.5:
                    child.set_facecolor(viridis_yellow)
                    child.set_edgecolor(viridis_yellow)
                # 蓝色系（负向）→ 紫色
                elif facecolor[0] < 0.5 and facecolor[1] < 0.5 and facecolor[2] > 0.5:
                    child.set_facecolor(viridis_purple)
                    child.set_edgecolor(viridis_purple)
    
    # 修改文本颜色，包括"higher"和"lower"
    for text in ax.texts:
        color = text.get_color()
        text_content = text.get_text()
        
        # 检查是否是"higher"或"lower"文本
        if "higher" in text_content.lower():
            text.set_color(viridis_yellow)  # 设置为黄色
        elif "lower" in text_content.lower():
            text.set_color(viridis_purple)  # 设置为紫色
        else:
            # 其他文本按原来的逻辑处理
            if isinstance(color, str):
                if color in ['red', 'r', '#ff0000']:
                    text.set_color(viridis_yellow)
                elif color in ['blue', 'b', '#0000ff']:
                    text.set_color(viridis_purple)
            elif hasattr(color, '__len__') and len(color) >= 3:
                if color[0] > 0.5 and color[1] < 0.5 and color[2] < 0.5:
                    text.set_color(viridis_yellow)
                elif color[0] < 0.5 and color[1] < 0.5 and color[2] > 0.5:
                    text.set_color(viridis_purple)
    
    # 修改箭头颜色
    for artist in ax.get_children():
        if hasattr(artist, 'get_facecolor'):
            facecolor = artist.get_facecolor()
            if hasattr(facecolor, '__len__') and len(facecolor) >= 3:
                # 红色箭头改为黄色
                if facecolor[0] > 0.5 and facecolor[1] < 0.5 and facecolor[2] < 0.5:
                    artist.set_facecolor(viridis_yellow)
                    if hasattr(artist, 'set_edgecolor'):
                        artist.set_edgecolor(viridis_yellow)
                # 蓝色箭头改为紫色
                elif facecolor[0] < 0.5 and facecolor[1] < 0.5 and facecolor[2] > 0.5:
                    artist.set_facecolor(viridis_purple)
                    if hasattr(artist, 'set_edgecolor'):
                        artist.set_edgecolor(viridis_purple)
    
    # === 关键修改1：恢复和美化x轴 ===
    
    # 显示x轴刻度和标签
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    # 设置x轴的刻度
    xlim = ax.get_xlim()
    x_ticks = np.linspace(xlim[0], xlim[1], 8)  # 设置8个刻度点
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{tick:.2f}' for tick in x_ticks])
    
    # 添加x轴标签
    ax.set_xlabel('Model Output Value', fontsize=12, fontweight='bold')
    
    # === 关键修改2：添加渐变色带（模拟热度条效果）===
    
    # 在图的顶部添加一个渐变色带来模拟热度条效果
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # 创建viridis渐变
    colors = ['#440154', '#31688e', '#35b779', '#fde725']  # viridis色系的关键颜色
    n_bins = 100
    viridis_cmap = LinearSegmentedColormap.from_list('viridis', colors, N=n_bins)
    
    # 在图的顶部添加颜色条
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    
    # 获取当前图形的位置
    pos = ax.get_position()
    
    # 创建颜色条的位置
    colorbar_ax = plt.gcf().add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.03])
    colorbar_ax.imshow(gradient, aspect='auto', cmap=viridis_cmap)
    colorbar_ax.set_xlim(0, 256)
    colorbar_ax.set_xticks([])
    colorbar_ax.set_yticks([])
    
    # 添加"higher"和"lower"标签到颜色条
    colorbar_ax.text(256*0.9, 0.5, 'higher', ha='center', va='center', 
                    color=viridis_yellow, fontsize=10, fontweight='bold')
    colorbar_ax.text(256*0.1, 0.5, 'lower', ha='center', va='center', 
                    color=viridis_purple, fontsize=10, fontweight='bold')
    
    # 添加箭头指向
    colorbar_ax.annotate('', xy=(256*0.85, 0.5), xytext=(256*0.95, 0.5),
                        arrowprops=dict(arrowstyle='->', color=viridis_yellow, lw=2))
    colorbar_ax.annotate('', xy=(256*0.15, 0.5), xytext=(256*0.05, 0.5),
                        arrowprops=dict(arrowstyle='->', color=viridis_purple, lw=2))
    
    # 设置主图的边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_visible(False)
    
    # 设置背景色
    ax.set_facecolor('white')
    
    # 设置标题
    plt.suptitle('SHAP Force Plot (Viridis Color Scheme)', fontsize=14, fontweight='bold', y=0.95)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis_formatted.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已生成格式完善的viridis色系SHAP force plot")
    
    # === 备选方案：如果上面的效果不理想，使用这个更简单的版本 ===
    
    plt.figure(figsize=(16, 8))
    
    # 重新绘制
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    ax = plt.gca()
    
    # 确保x轴可见
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    
    # 设置x轴格式
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlabel('Model Output', fontsize=12, fontweight='bold')
    
    # 添加网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # 设置边框
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.title('SHAP Force Plot - Alternative Format', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_alternative.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已生成备选格式的SHAP force plot")

    # --- HTML 版本保持不变 ---
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns),
    )

    # HTML版本使用PkYg颜色方案
    shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], 
                               plot_cmap="PkYg")
    
    save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.html"
    shap.save_html(save_path, shap_plot)

    import webbrowser
    webbrowser.open(save_path)
    
else:
    print("指定的特征未在 X_test 中找到。")