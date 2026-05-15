import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data_path = os.path.join("data", "processed", "features_multi_seasons.csv")
    if not os.path.exists(data_path):
        print(f"找不到特征数据 {data_path}，请先运行 feature_engineering.py。")
        return
        
    print(f"正在加载特征数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. 筛选数值型特征列和预测目标列
    numeric_df = df.select_dtypes(include=['number']).copy()
    
    # 还需要移除不在分析范围内的目标列和 ID 类的无意义数值列
    cols_to_drop = ['GAME_ID', 'TEAM_ID_HOME', 'TEAM_ID_AWAY', 'TARGET_PLUS_MINUS_HOME']
    numeric_df = numeric_df.drop(columns=[col for col in cols_to_drop if col in numeric_df.columns])
    
    print("正在计算特征的皮尔逊相关系数矩阵...")
    # 计算相关系数 (Pearson Correlation)
    corr_matrix = numeric_df.corr()
    
    # 提取出与分类预测目标 (TARGET_WIN_HOME) 相关的系数，并降序打印出来，方便直接查看
    target_corr = corr_matrix['TARGET_WIN_HOME'].sort_values(ascending=False)
    print("\n========== 特征与【主队胜负(TARGET_WIN_HOME)】的相关性排名 ==========")
    print(target_corr)
    print("===================================================================\n")

    print("正在绘制热力图并保存为图片...")
    # 2. 设置画板大小 (16x12 可以保证文字清晰不拥挤)
    plt.figure(figsize=(16, 12))
    
    # 3. 使用 seaborn 绘制热力图
    # cmap="coolwarm": 蓝-白-红 渐变色，红色越深代表正相关越强，蓝色越深代表负相关越强
    # annot=True: 在格子里显示具体数字，fmt=".2f" 保留两位小数
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                vmin=-1, vmax=1, center=0, 
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .8})
                
    # 调整标题和布局
    plt.title('NBA Game Features Correlation Heatmap (2023-24 Season)', fontsize=18, pad=20)
    plt.tight_layout()
    
    # 4. 保存为高分辨率图片
    output_img = "correlation_heatmap.png"
    plt.savefig(output_img, dpi=300)
    print(f"🎉 热力图绘制成功！已保存至项目根目录: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    main()
