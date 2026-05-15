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
    
    # 2. 抓取数据
    season = "2023-24" # 你可以修改为你想抓取的赛季，或者使用循环抓取多个赛季
    df_games = fetch_season_games(season)
    
    # 3. 数据预览与保存
    output_path = os.path.join(raw_data_dir, f"raw_games_{season.replace('-', '_')}.csv")
    df_games.to_csv(output_path, index=False)
    
    print(f"\n抓取成功！共获取 {len(df_games)} 条比赛记录。")
    print(f"原始数据已保存至: {output_path}\n")
    
    print("数据前 5 行预览:")
    print(df_games[['GAME_DATE', 'MATCHUP', 'WL', 'PTS']].head())

if __name__ == "__main__":
    main()
