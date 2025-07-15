# 完全修色解决方案 - 确保所有元素都变成viridis色系

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
    
    # 定义viridis颜色
    viridis_yellow = "#fde725"  # 亮黄色 - 正向影响
    viridis_purple = "#440154"  # 深紫色 - 负向影响
    
    plt.figure(figsize=(16, 7))
    
    # 绘制原始的matplotlib版本
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    ax = plt.gca()
    
    # === 全面的颜色修改 ===
    
    import matplotlib.patches as patches
    import matplotlib.lines as mlines
    
    # 1. 修改所有矩形元素（包括热度条和其他矩形）
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
                # 检查其他可能的蓝色变体
                elif facecolor[2] > 0.6:  # 任何偏蓝的颜色
                    child.set_facecolor(viridis_purple)
                    child.set_edgecolor(viridis_purple)
    
    # 2. 修改所有多边形和路径（包括箭头）
    for child in ax.get_children():
        if isinstance(child, patches.Polygon):
            facecolor = child.get_facecolor()
            if len(facecolor) >= 3:
                # 红色系箭头改为黄色
                if facecolor[0] > 0.5 and facecolor[1] < 0.5 and facecolor[2] < 0.5:
                    child.set_facecolor(viridis_yellow)
                    child.set_edgecolor(viridis_yellow)
                # 蓝色系箭头改为紫色
                elif facecolor[0] < 0.5 and facecolor[1] < 0.5 and facecolor[2] > 0.5:
                    child.set_facecolor(viridis_purple)
                    child.set_edgecolor(viridis_purple)
                # 检查其他蓝色变体
                elif facecolor[2] > 0.6:
                    child.set_facecolor(viridis_purple)
                    child.set_edgecolor(viridis_purple)
    
    # 3. 修改所有FancyBboxPatch和其他patch类型
    for child in ax.get_children():
        if isinstance(child, (patches.FancyBboxPatch, patches.PathPatch)):
            facecolor = child.get_facecolor()
            if len(facecolor) >= 3:
                # 红色系改为黄色
                if facecolor[0] > 0.5 and facecolor[1] < 0.5 and facecolor[2] < 0.5:
                    child.set_facecolor(viridis_yellow)
                    child.set_edgecolor(viridis_yellow)
                # 蓝色系改为紫色
                elif facecolor[0] < 0.5 and facecolor[1] < 0.5 and facecolor[2] > 0.5:
                    child.set_facecolor(viridis_purple)
                    child.set_edgecolor(viridis_purple)
                # 检查其他蓝色变体
                elif facecolor[2] > 0.6:
                    child.set_facecolor(viridis_purple)
                    child.set_edgecolor(viridis_purple)
    
    # 4. 修改所有线条（包括箭头线条）
    for child in ax.get_children():
        if isinstance(child, mlines.Line2D):
            color = child.get_color()
            if isinstance(color, str):
                if color in ['red', 'r', '#ff0000']:
                    child.set_color(viridis_yellow)
                elif color in ['blue', 'b', '#0000ff']:
                    child.set_color(viridis_purple)
            elif hasattr(color, '__len__') and len(color) >= 3:
                if color[0] > 0.5 and color[1] < 0.5 and color[2] < 0.5:
                    child.set_color(viridis_yellow)
                elif color[0] < 0.5 and color[1] < 0.5 and color[2] > 0.5:
                    child.set_color(viridis_purple)
                elif color[2] > 0.6:
                    child.set_color(viridis_purple)
    
    # 5. 修改所有文本颜色
    for text in ax.texts:
        text_content = text.get_text().lower()
        
        # 专门处理"higher"和"lower"文本
        if "higher" in text_content or "high" in text_content:
            text.set_color(viridis_yellow)
        elif "lower" in text_content or "low" in text_content:
            text.set_color(viridis_purple)
        else:
            # 其他文本的颜色处理
            color = text.get_color()
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
                elif color[2] > 0.6:  # 任何偏蓝的颜色
                    text.set_color(viridis_purple)
    
    # 6. 修改集合对象（Collections）
    for collection in ax.collections:
        if hasattr(collection, 'get_facecolors'):
            facecolors = collection.get_facecolors()
            if len(facecolors) > 0:
                new_facecolors = []
                for color in facecolors:
                    if len(color) >= 3:
                        # 红色系改为黄色
                        if color[0] > 0.5 and color[1] < 0.5 and color[2] < 0.5:
                            new_facecolors.append(viridis_yellow)
                        # 蓝色系改为紫色
                        elif color[0] < 0.5 and color[1] < 0.5 and color[2] > 0.5:
                            new_facecolors.append(viridis_purple)
                        # 检查其他蓝色变体
                        elif color[2] > 0.6:
                            new_facecolors.append(viridis_purple)
                        else:
                            new_facecolors.append(color)
                    else:
                        new_facecolors.append(color)
                collection.set_facecolors(new_facecolors)
        
        if hasattr(collection, 'get_edgecolors'):
            edgecolors = collection.get_edgecolors()
            if len(edgecolors) > 0:
                new_edgecolors = []
                for color in edgecolors:
                    if len(color) >= 3:
                        if color[0] > 0.5 and color[1] < 0.5 and color[2] < 0.5:
                            new_edgecolors.append(viridis_yellow)
                        elif color[0] < 0.5 and color[1] < 0.5 and color[2] > 0.5:
                            new_edgecolors.append(viridis_purple)
                        elif color[2] > 0.6:
                            new_edgecolors.append(viridis_purple)
                        else:
                            new_edgecolors.append(color)
                    else:
                        new_edgecolors.append(color)
                collection.set_edgecolors(new_edgecolors)
    
    # 7. 特殊处理：强制查找并修改所有可能的蓝色元素
    def force_change_blue_elements(ax):
        """强制修改所有蓝色元素"""
        for child in ax.get_children():
            # 检查所有可能的颜色属性
            color_attrs = ['get_facecolor', 'get_edgecolor', 'get_color']
            for attr_name in color_attrs:
                if hasattr(child, attr_name):
                    try:
                        color = getattr(child, attr_name)()
                        if color is not None:
                            if isinstance(color, str):
                                if 'blue' in color.lower() or color in ['b', '#0000ff']:
                                    if hasattr(child, attr_name.replace('get_', 'set_')):
                                        getattr(child, attr_name.replace('get_', 'set_'))(viridis_purple)
                            elif hasattr(color, '__len__') and len(color) >= 3:
                                # 检查是否为蓝色系
                                if (color[2] > 0.6 and color[0] < 0.5 and color[1] < 0.5) or color[2] > 0.7:
                                    if hasattr(child, attr_name.replace('get_', 'set_')):
                                        getattr(child, attr_name.replace('get_', 'set_'))(viridis_purple)
                    except:
                        continue
    
    # 执行强制蓝色修改
    force_change_blue_elements(ax)
    
    # 8. 显示x轴和美化
    ax.xaxis.set_visible(True)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlabel('Model Output Value', fontsize=12, fontweight='bold')
    
    # 设置边框
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置背景色
    ax.set_facecolor('white')
    
    # 设置标题
    plt.title('SHAP Force Plot (Complete Viridis Color Scheme)', fontsize=14, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_complete_viridis.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已完成全面的viridis色系修改，包括箭头、负相关区域和所有蓝色元素")
    
    # === 备选方案：如果上面还是不够，使用更暴力的方法 ===
    
    print("\n生成备选方案...")
    
    plt.figure(figsize=(16, 7))
    
    # 重新绘制
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    ax = plt.gca()
    
    # 暴力方法：遍历所有子元素并修改所有可能的颜色属性
    def brutal_color_change(element):
        """暴力修改所有颜色属性"""
        methods_to_try = [
            ('set_facecolor', viridis_purple),
            ('set_edgecolor', viridis_purple),
            ('set_color', viridis_purple),
            ('set_markerfacecolor', viridis_purple),
            ('set_markeredgecolor', viridis_purple),
        ]
        
        for method_name, color in methods_to_try:
            if hasattr(element, method_name):
                try:
                    # 检查当前颜色是否为蓝色系
                    current_color = None
                    get_method = method_name.replace('set_', 'get_')
                    if hasattr(element, get_method):
                        current_color = getattr(element, get_method)()
                    
                    if current_color is not None:
                        if isinstance(current_color, str):
                            if 'blue' in current_color.lower() or current_color in ['b', '#0000ff']:
                                getattr(element, method_name)(color)
                        elif hasattr(current_color, '__len__') and len(current_color) >= 3:
                            if current_color[2] > 0.6 and current_color[0] < 0.5:
                                getattr(element, method_name)(color)
                except:
                    continue
    
    # 对所有元素应用暴力修改
    for child in ax.get_children():
        brutal_color_change(child)
    
    # 显示x轴
    ax.xaxis.set_visible(True)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlabel('Model Output Value', fontsize=12, fontweight='bold')
    
    plt.title('SHAP Force Plot (Brutal Color Change)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_brutal_fix.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("已生成备选方案，使用更强制的颜色修改方法")

    # --- HTML 版本 ---
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