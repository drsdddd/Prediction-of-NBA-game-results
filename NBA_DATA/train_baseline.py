import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings

# 忽略一些 sklearn 的无害警告
warnings.filterwarnings('ignore')

def main():
    data_path = os.path.join("data", "processed", "features_multi_seasons.csv")
    if not os.path.exists(data_path):
        print(f"数据不存在: {data_path}，请先运行特征工程脚本。")
        return
        
    print(f"正在加载数据: {data_path}...")
    df = pd.read_csv(data_path)
    
    # 【核心细节】：在体育预测中，千万不能用随机打乱 (shuffle) 来切分数据集！
    # 否则你会用“未来的数据”去训练，然后预测“过去的数据”，导致准确率虚高。
    # 我们必须按时间排序，用前几年的数据训练，预测最近两年的比赛。
    df = df.sort_values('GAME_DATE')
    
    # 筛选特征列：所有包含 '5G_AVG' 的数值列（包含了 HOME, AWAY 以及我们自己造的 DIFF 差值特征）
    # 以及新加入的超强动态特征 'ELO'
    feature_cols = [col for col in df.columns if '5G_AVG' in col or 'ELO' in col]
    
    # 准备 X 和 y
    X = df[feature_cols].fillna(0) # 兜底防止意外出现缺失值
    y = df['TARGET_WIN_HOME']
    
    # 取前 80% 比赛作为训练集，最近 20% 的比赛作为测试集 (且保持时间顺序)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"\n数据集切分完毕！")
    print(f"训练集 (过去): {len(X_train)} 场比赛")
    print(f"测试集 (未来): {len(X_test)} 场比赛")
    
    print("\n================= 模型 1: 逻辑回归 (Logistic Regression) =================")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"-> 逻辑回归 预测准确率 (Accuracy): {acc_lr:.4f} ({(acc_lr*100):.2f}%)")
    
    print("\n================= 模型 2: 随机森林 (Random Forest) =======================")
    # 限制树的深度防止过拟合
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"-> 随机森林 预测准确率 (Accuracy): {acc_rf:.4f} ({(acc_rf*100):.2f}%)")
    
    print("\n[随机森林 - 详细评估报告]")
    print(classification_report(y_test, y_pred_rf, target_names=['客队赢 (0)', '主队赢 (1)']))
    
    print("\n[随机森林 - 特征重要性 TOP 10] (算法认为最有用的指标)")
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for index, val in importances.head(10).items():
        print(f"{index:<25} : {val:.4f}")

if __name__ == "__main__":
    main()
