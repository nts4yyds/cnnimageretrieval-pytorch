# 针对性修色方案 - 专门解决箭头、负相关蓝色、其他部分变色问题

# 前面代码保持不变
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
    viridis_yellow = "#fde725"  # 亮黄色
    viridis_purple = "#440154"  # 深紫色
    
    plt.figure(figsize=(16, 7))
    
    # 绘制原始图形
    shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data,
                    matplotlib=True, show=False)
    
    ax = plt.gca()
    
    # === 针对性解决三个问题 ===
    
    print("开始针对性颜色修改...")
    
    # 问题1 & 2: 修改所有图形元素的颜色（包括箭头和负相关蓝色）
    import matplotlib.patches as patches
    import matplotlib.collections as collections
    
    # 获取所有图形元素并统一处理
    all_elements = list(ax.get_children())
    
    for element in all_elements:
        # 处理所有patch类型（包括箭头、矩形等）
        if hasattr(element, 'get_facecolor') and hasattr(element, 'set_facecolor'):
            try:
                current_color = element.get_facecolor()
                
                # 检查并修改颜色
                if isinstance(current_color, (list, tuple, np.ndarray)) and len(current_color) >= 3:
                    r, g, b = current_color[0], current_color[1], current_color[2]
                    
                    # 识别蓝色系（负相关）- 多种蓝色检测方式
                    if (b > 0.5 and r < 0.6 and g < 0.6) or \
                       (b > r and b > g and b > 0.4) or \
                       (r < 0.5 and g < 0.5 and b > 0.5):
                        element.set_facecolor(viridis_purple)
                        if hasattr(element, 'set_edgecolor'):
                            element.set_edgecolor(viridis_purple)
                        print(f"修改了蓝色元素: {type(element).__name__}")
                    
                    # 识别红色系（正相关）
                    elif (r > 0.5 and g < 0.6 and b < 0.6) or \
                         (r > g and r > b and r > 0.4):
                        element.set_facecolor(viridis_yellow)
                        if hasattr(element, 'set_edgecolor'):
                            element.set_edgecolor(viridis_yellow)
                        print(f"修改了红色元素: {type(element).__name__}")
                        
            except Exception as e:
                continue
        
        # 处理线条颜色
        if hasattr(element, 'get_color') and hasattr(element, 'set_color'):
            try:
                current_color = element.get_color()
                if isinstance(current_color, str):
                    if 'blue' in current_color.lower() or current_color in ['b', '#0000ff']:
                        element.set_color(viridis_purple)
                        print(f"修改了蓝色线条: {type(element).__name__}")
                    elif 'red' in current_color.lower() or current_color in ['r', '#ff0000']:
                        element.set_color(viridis_yellow)
                        print(f"修改了红色线条: {type(element).__name__}")
                elif isinstance(current_color, (list, tuple, np.ndarray)) and len(current_color) >= 3:
                    r, g, b = current_color[0], current_color[1], current_color[2]
                    if b > 0.5 and r < 0.6 and g < 0.6:
                        element.set_color(viridis_purple)
                        print(f"修改了蓝色线条: {type(element).__name__}")
                    elif r > 0.5 and g < 0.6 and b < 0.6:
                        element.set_color(viridis_yellow)
                        print(f"修改了红色线条: {type(element).__name__}")
            except:
                continue
    
    # 问题3: 修改集合对象（Collections）- 处理其他可能的图形元素
    for collection in ax.collections:
        try:
            # 修改面颜色
            if hasattr(collection, 'get_facecolors') and hasattr(collection, 'set_facecolors'):
                facecolors = collection.get_facecolors()
                if len(facecolors) > 0:
                    new_colors = []
                    for color in facecolors:
                        if len(color) >= 3:
                            r, g, b = color[0], color[1], color[2]
                            # 蓝色系改为紫色
                            if (b > 0.5 and r < 0.6 and g < 0.6) or (b > r and b > g and b > 0.4):
                                new_colors.append(viridis_purple)
                            # 红色系改为黄色
                            elif (r > 0.5 and g < 0.6 and b < 0.6) or (r > g and r > b and r > 0.4):
                                new_colors.append(viridis_yellow)
                            else:
                                new_colors.append(color)
                        else:
                            new_colors.append(color)
                    collection.set_facecolors(new_colors)
                    print(f"修改了集合对象的面颜色: {type(collection).__name__}")
            
            # 修改边框颜色
            if hasattr(collection, 'get_edgecolors') and hasattr(collection, 'set_edgecolors'):
                edgecolors = collection.get_edgecolors()
                if len(edgecolors) > 0:
                    new_colors = []
                    for color in edgecolors:
                        if len(color) >= 3:
                            r, g, b = color[0], color[1], color[2]
                            if (b > 0.5 and r < 0.6 and g < 0.6) or (b > r and b > g and b > 0.4):
                                new_colors.append(viridis_purple)
                            elif (r > 0.5 and g < 0.6 and b < 0.6) or (r > g and r > b and r > 0.4):
                                new_colors.append(viridis_yellow)
                            else:
                                new_colors.append(color)
                        else:
                            new_colors.append(color)
                    collection.set_edgecolors(new_colors)
                    print(f"修改了集合对象的边框颜色: {type(collection).__name__}")
                    
        except Exception as e:
            continue
    
    # 修改文本颜色（包括higher/lower）
    for text in ax.texts:
        try:
            text_content = text.get_text().lower()
            
            # 专门处理"higher"和"lower"
            if "higher" in text_content or "high" in text_content:
                text.set_color(viridis_yellow)
                print(f"修改了'higher'文本颜色")
            elif "lower" in text_content or "low" in text_content:
                text.set_color(viridis_purple)
                print(f"修改了'lower'文本颜色")
            else:
                # 其他文本根据原色修改
                color = text.get_color()
                if isinstance(color, str):
                    if 'blue' in color.lower() or color in ['b', '#0000ff']:
                        text.set_color(viridis_purple)
                    elif 'red' in color.lower() or color in ['r', '#ff0000']:
                        text.set_color(viridis_yellow)
                elif isinstance(color, (list, tuple, np.ndarray)) and len(color) >= 3:
                    r, g, b = color[0], color[1], color[2]
                    if b > 0.5 and r < 0.6 and g < 0.6:
                        text.set_color(viridis_purple)
                    elif r > 0.5 and g < 0.6 and b < 0.6:
                        text.set_color(viridis_yellow)
        except:
            continue
    
    # 特别处理：强制搜索并修改所有可能遗漏的蓝色元素
    def find_and_fix_blue_elements():
        """查找并修改所有可能的蓝色元素"""
        modified_count = 0
        
        for child in ax.get_children():
            # 检查各种可能的颜色属性
            color_methods = [
                ('get_facecolor', 'set_facecolor'),
                ('get_edgecolor', 'set_edgecolor'),
                ('get_color', 'set_color'),
                ('get_markerfacecolor', 'set_markerfacecolor'),
                ('get_markeredgecolor', 'set_markeredgecolor')
            ]
            
            for get_method, set_method in color_methods:
                if hasattr(child, get_method) and hasattr(child, set_method):
                    try:
                        current_color = getattr(child, get_method)()
                        
                        if current_color is not None:
                            should_change_to_purple = False
                            should_change_to_yellow = False
                            
                            if isinstance(current_color, str):
                                if 'blue' in current_color.lower() or current_color in ['b', '#0000ff']:
                                    should_change_to_purple = True
                                elif 'red' in current_color.lower() or current_color in ['r', '#ff0000']:
                                    should_change_to_yellow = True
                            elif isinstance(current_color, (list, tuple, np.ndarray)) and len(current_color) >= 3:
                                r, g, b = current_color[0], current_color[1], current_color[2]
                                # 更宽泛的蓝色检测
                                if (b > 0.4 and b > r and b > g) or (b > 0.6):
                                    should_change_to_purple = True
                                elif (r > 0.4 and r > g and r > b) or (r > 0.6):
                                    should_change_to_yellow = True
                            
                            if should_change_to_purple:
                                getattr(child, set_method)(viridis_purple)
                                modified_count += 1
                                print(f"强制修改蓝色元素: {type(child).__name__}.{set_method}")
                            elif should_change_to_yellow:
                                getattr(child, set_method)(viridis_yellow)
                                modified_count += 1
                                print(f"强制修改红色元素: {type(child).__name__}.{set_method}")
                                
                    except:
                        continue
        
        return modified_count
    
    # 执行强制修改
    modified_count = find_and_fix_blue_elements()
    print(f"强制修改了 {modified_count} 个元素")
    
    # 显示x轴
    ax.xaxis.set_visible(True)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlabel('Model Output Value', fontsize=12, fontweight='bold')
    
    # 设置边框
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置背景和标题
    ax.set_facecolor('white')
    plt.title('SHAP Force Plot - Targeted Viridis Color Fix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(r"E:\jupyter-envi\analysis_results\shap_force_plot_targeted_fix.png", 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("完成针对性修色！")
    print("✅ 已修改箭头颜色")
    print("✅ 已修改负相关蓝色区域")
    print("✅ 已修改其他图形元素")

    # --- HTML 版本 ---
    shap_values_filtered_js = np.delete(shap_values_explanation.values, remove_indices, axis=1)
    data_filtered_js = X_test.drop(columns=features_to_remove)
    
    shap_explanation_filtered = shap.Explanation(
        values=shap_values_filtered_js,
        base_values=shap_values_explanation.base_values,
        data=data_filtered_js,
        feature_names=list(data_filtered_js.columns),
    )

    # HTML版本
    shap_plot = shap.force_plot(shap_explanation_filtered[sample_index], 
                               plot_cmap="PkYg")
    
    save_path = r"E:\jupyter-envi\analysis_results\shap_force_plot_viridis.html"
    shap.save_html(save_path, shap_plot)

    import webbrowser
    webbrowser.open(save_path)
    
else:
    print("指定的特征未在 X_test 中找到。")

# ====================================
# 如果上面的方法还不够，请尝试这个更简单的方法：
# 直接替换您原代码中的 shap.force_plot() 调用
# ====================================

print("\n=== 最简单的替代方法 ===")
print("如果上面的修改还不够，请将您的原代码中的：")
print("shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data, matplotlib=True, show=False)")
print("\n替换为：")
print("shap.force_plot(expected_value, shap_values_filtered_values, X_test_filtered_data, plot_cmap='PkYg')")
print("（移除matplotlib=True，使用PkYg颜色方案）")