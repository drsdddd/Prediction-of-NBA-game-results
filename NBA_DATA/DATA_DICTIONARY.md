# NBA 比赛预测项目 - 数据特征对照表 (Data Dictionary)

这份文档旨在帮助团队成员理解 `features_multi_seasons.csv` 数据集中每一列的含义。
数据经过处理后，每一行代表**一场完整的比赛**，特征被划分为主队（后缀 `_HOME`）、客队（后缀 `_AWAY`）以及两队之间的差值（前缀 `DIFF_`）。

## 1. 基础信息列 (Base Columns)
| 字段名 (以主队为例) | 含义 | 说明 |
| :--- | :--- | :--- |
| `GAME_ID` | 比赛唯一识别码 | NBA 官方分配的 10 位数比赛 ID |
| `GAME_DATE` | 比赛日期 | 格式为 YYYY-MM-DD |
| `TEAM_ID_HOME` / `_AWAY` | 球队识别码 | NBA 官方分配的球队 ID |
| `TEAM_ABBREVIATION_HOME` / `_AWAY`| 球队缩写 | 如 LAL(湖人), BOS(凯尔特人), GSW(勇士) 等 |
| `SEASON_ID` | 赛季识别码 | 以 '2' 开头代表常规赛，例如 '22023' 代表 23-24 赛季常规赛 |

## 2. 动态实力评估特征 (ELO Rating) 👑 核心特征
这是通过模拟著名数据网站 FiveThirtyEight 的算法，跨越 10 个赛季逐场计算出的高级能力指标（包含了跨赛季 25% 均值回归机制）。
| 字段名 | 业务含义 / 解释 |
| :--- | :--- |
| `ELO_HOME` | 主队在赛前的 ELO 等级分 |
| `ELO_AWAY` | 客队在赛前的 ELO 等级分 |
| `DIFF_ELO` | 两队 ELO 分差 (`ELO_HOME` - `ELO_AWAY`)，正数代表主队纸面实力更强 |

## 3. 近期滚动状态特征 (Rolling Features)
以下所有特征都带有 `_5G_AVG` 标识，代表基于赛前**最近 5 场比赛**的滚动平均值。分为 `_HOME` (主队数据)、`_AWAY` (客队数据) 以及 `DIFF_` (主客队差值：主队 - 客队)。

| 基础字段名 (以主队为例) | 对应差值特征 | 业务含义 / 解释 |
| :--- | :--- | :--- |
| `PLUS_MINUS_5G_AVG_HOME`| `DIFF_PLUS_MINUS_5G_AVG`| 近 5 场平均净胜分，反映近期综合统治力 |
| `WIN_5G_AVG_HOME` | `DIFF_WIN_5G_AVG` | 近 5 场胜率 (范围 0.0 ~ 1.0)，反映近期竞技状态 |
| `PTS_5G_AVG_HOME` | `DIFF_PTS_5G_AVG` | 近 5 场平均得分，反映近期进攻火力 |
| `FG_PCT_5G_AVG_HOME` | `DIFF_FG_PCT_5G_AVG` | 近 5 场投篮命中率，反映近期投射效率 |
| `FG3_PCT_5G_AVG_HOME` | `DIFF_FG3_PCT_5G_AVG` | 近 5 场三分球命中率，反映外线威胁 |
| `FT_PCT_5G_AVG_HOME` | `DIFF_FT_PCT_5G_AVG` | 近 5 场罚球命中率，反映基本功 |
| `REB_5G_AVG_HOME` | `DIFF_REB_5G_AVG` | 近 5 场平均篮板数，反映内线掌控力 |
| `AST_5G_AVG_HOME` | `DIFF_AST_5G_AVG` | 近 5 场平均助攻数，反映团队配合 |
| `TOV_5G_AVG_HOME` | `DIFF_TOV_5G_AVG` | 近 5 场平均失误数，反映控球稳定性 (越低越好) |

## 4. 模型预测目标列 (Target Columns)
这两个字段是留给机器学习模型进行“对答案”（训练）用的标签，包含了真实的比赛结果。

| 字段名 (预测目标) | 含义 | 说明 |
| :--- | :--- | :--- |
| `TARGET_WIN_HOME` | 主队是否获胜 (分类) | `1` 代表主队赢球，`0` 代表主队输球 (用于逻辑回归、随机森林等**分类任务**) |
| `TARGET_PLUS_MINUS_HOME`| 主队净胜分 (回归) | 主队得分 - 客队得分。如为 `10` 表示主队赢10分，`-5` 表示主队输5分 (用于**回归任务**评估) |
