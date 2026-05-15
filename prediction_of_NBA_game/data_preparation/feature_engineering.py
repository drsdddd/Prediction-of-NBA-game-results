import os
import pandas as pd
import numpy as np

def load_data(filepath):
    print(f"正在加载数据: {filepath}...")
    df = pd.read_csv(filepath)
    # 将日期字符串转换为 datetime 对象，方便后续按时间排序
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

def clean_data(df):
    """
    基础清洗：处理缺失值，构建基础标识字段。
    """
    # 丢弃没有胜负结果的无效比赛数据（比如未来未打的比赛或数据错误）
    df = df.dropna(subset=['WL']).copy()
    
    # 1. 胜负标签二进制化 (Win=1, Loss=0)
    df['WIN'] = df['WL'].map({'W': 1, 'L': 0})
    
    # 2. 提取主客场标识 (MATCHUP 中包含 'vs.' 的为主场，'@' 为客场)
    df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
    
    return df

def create_rolling_features(df, window=5):
    """
    核心：为每支球队计算基于历史表现的滚动特征 (如近 5 场胜率、场均得分等)。
    【防穿越（数据泄露）机制】：使用 shift(1)，确保预测今天比赛时，只用到昨天之前的统计数据。
    """
    print(f"正在计算滚动统计特征 (窗口大小={window}场)...")
    
    # 必须先按球队和比赛日期排序
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
    
    # 我们打算做平滑/滚动平均的统计列
    stats_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 'PLUS_MINUS', 'WIN']
    
    # 分组计算
    grouped = df.groupby('TEAM_ID')
    
    # 这里是精髓：先分组，再取出统计列，接着 shift(1) 把所有数据往后推一场，然后再算 rolling mean
    # min_periods=1 允许赛季初的比赛也能算出平均（比如第2场比赛，只有第1场的数据作为平均）
    rolling_stats = grouped[stats_cols].apply(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    
    # 重命名列名，加上 _5G_AVG 后缀以示区分
    rolling_cols = {col: f'{col}_{window}G_AVG' for col in stats_cols}
    rolling_stats = rolling_stats.rename(columns=rolling_cols)
    
    # 把计算出来的特征拼接到原数据后面
    df = pd.concat([df, rolling_stats], axis=1)
    
    # 赛季第一场比赛没有历史数据 (shift(1) 会产生 NaN)，我们直接丢掉它们
    df = df.dropna(subset=[f'WIN_{window}G_AVG'])
    
    return df

def create_matchup_dataset(df):
    """
    将原始“球队视角”的一行行数据，整合成“比赛视角”（即主队 vs 客队在同一行），
    这是机器学习模型标准的数据输入格式。
    """
    print("正在构建对阵数据集 (主队 vs 客队)...")
    
    # 拆分主场队和客场队
    home_df = df[df['IS_HOME'] == 1].copy()
    away_df = df[df['IS_HOME'] == 0].copy()
    
    # 筛选我们需要保留在最终训练集里的特征列
    feature_cols = [col for col in df.columns if col.endswith('G_AVG')]
    base_cols = ['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION']
    
    # 主队还需要保留胜负和分差，因为这会作为我们预测的目标变量 (Target)
    home_df = home_df[base_cols + feature_cols + ['WIN', 'PLUS_MINUS']]
    away_df = away_df[base_cols + feature_cols]
    
    # 给列名加上后缀，防止 merge 时重名
    home_df = home_df.add_suffix('_HOME')
    away_df = away_df.add_suffix('_AWAY')
    
    # GAME_ID 被改名了，改回来以便当作 Join 的 Key，GAME_DATE 也一样
    home_df = home_df.rename(columns={'GAME_ID_HOME': 'GAME_ID', 'GAME_DATE_HOME': 'GAME_DATE'})
    away_df = away_df.rename(columns={'GAME_ID_AWAY': 'GAME_ID'})
    
    # 按比赛 ID 内连接
    matchup_df = pd.merge(home_df, away_df, on='GAME_ID', how='inner')
    
    # 构建最终的模型预测目标
    matchup_df['TARGET_WIN_HOME'] = matchup_df['WIN_HOME']               # 分类任务 Target：主队是否获胜
    matchup_df['TARGET_PLUS_MINUS_HOME'] = matchup_df['PLUS_MINUS_HOME'] # 回归任务 Target：主队胜负分差
    
    # 删掉作为中间产物保留的列
    matchup_df = matchup_df.drop(columns=['WIN_HOME', 'PLUS_MINUS_HOME'])
    
    return matchup_df

def main():
    input_path = os.path.join("data", "raw", "raw_games_2023_24.csv")
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"找不到原始数据 {input_path}。请先运行 fetch_data.py！")
        return
        
    # 执行流水线
    df = load_data(input_path)
    df = clean_data(df)
    df = create_rolling_features(df, window=5)
    matchup_df = create_matchup_dataset(df)
    
    # 保存结果
    output_path = os.path.join(output_dir, "features_2023_24.csv")
    matchup_df.to_csv(output_path, index=False)
    
    print(f"\n特征工程已完成！")
    print(f"成功生成了 {len(matchup_df)} 场对阵特征数据。")
    print(f"特征集已保存至: {output_path}\n")
    
    print("模型输入特征集预览:")
    cols_to_show = ['GAME_DATE', 'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY', 
                    'WIN_5G_AVG_HOME', 'WIN_5G_AVG_AWAY', 'TARGET_WIN_HOME']
    print(matchup_df[cols_to_show].head())

if __name__ == "__main__":
    main()
