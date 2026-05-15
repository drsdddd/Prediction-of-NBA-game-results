import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
import time

def fetch_season_games(season="2023-24"):
    """
    使用 nba_api 抓取指定赛季的比赛数据。
    """
    print(f"正在抓取 {season} 赛季的比赛数据...")
    
    # LeagueGameFinder 可以获取联盟中所有的比赛记录
    # season_nullable 参数格式通常为 'YYYY-YY'，如 '2023-24'
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    
    # 返回的是一个包含 DataFrame 的列表，取第一个即可
    games_df = gamefinder.get_data_frames()[0]
    
    # nba_api 有请求频率限制，建议加入延时以防被封 IP
    time.sleep(1) 
    
    return games_df

def main():
    # 1. 创建数据存储目录结构
    raw_data_dir = os.path.join("data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # 2. 批量抓取近 10 个赛季的数据 (2014 ~ 2023)
    seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2014, 2024)]
    all_games_list = []
    
    for season in seasons:
        try:
            df = fetch_season_games(season)
            all_games_list.append(df)
        except Exception as e:
            print(f"抓取 {season} 失败: {e}")
            
    # 合并所有赛季数据
    df_games = pd.concat(all_games_list, ignore_index=True)
    
    # 3. 数据预览与保存
    output_path = os.path.join(raw_data_dir, "raw_games_multi_seasons.csv")
    df_games.to_csv(output_path, index=False)
    
    print(f"\n批量抓取成功！共获取 {len(seasons)} 个赛季, {len(df_games)} 条比赛记录。")
    print(f"原始数据已合并保存至: {output_path}\n")
    
    print("数据前 5 行预览:")
    print(df_games[['GAME_DATE', 'MATCHUP', 'WL', 'PTS']].head())

if __name__ == "__main__":
    main()
