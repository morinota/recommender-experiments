"""マイクロバッチによるバンディット学習の例."""

import numpy as np
import matplotlib.pyplot as plt

from recommender_experiments.service.algorithms.thompson_sampling_ranking import (
    ThompsonSamplingRanking,
)
from recommender_experiments.service.environment.ranking_synthetic_dataset import (
    RankingSyntheticBanditDataset,
)


def main():
    """マイクロバッチによるバンディット学習のデモンストレーション."""
    print("=== マイクロバッチ バンディット学習のデモ ===")

    # 1. 環境設定
    dim_context = 4
    num_actions = 8
    K = 3
    
    # パラメータを設定
    np.random.seed(42)
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, dim_context * 2))

    # 動的action変化を設定
    action_churn_schedule = {
        0: [0, 1, 2, 3, 4],        # 0-199試行目: action 0-4が利用可能
        200: [2, 3, 4, 5, 6],      # 200-399試行目: action 2-6が利用可能
        400: [4, 5, 6, 7],         # 400-599試行目: action 4-7が利用可能
    }

    # 2. データセットを作成
    dataset = RankingSyntheticBanditDataset(
        dim_context=dim_context,
        num_actions=num_actions,
        k=K,
        theta=theta,
        quadratic_weights=quadratic_weights,
        action_bias=action_bias,
        position_interaction_weights=position_interaction_weights,
        action_context=action_context,
        action_churn_schedule=action_churn_schedule,
        random_state=42,
    )

    # 3. バンディットアルゴリズムを初期化
    algorithm = ThompsonSamplingRanking(
        num_actions=num_actions, 
        k=K, 
        dim_context=dim_context,
        alpha=1.0, 
        beta=1.0, 
        random_state=42
    )

    # 4. マイクロバッチ学習シミュレーション
    num_trials = 600
    batch_size = 10  # マイクロバッチサイズ
    
    instant_regrets = []
    cumulative_regret = 0.0
    
    print(f"シミュレーション開始 - {num_trials} trials, batch size: {batch_size}")
    
    for trial in range(0, num_trials, batch_size):
        # マイクロバッチデータを生成（バンディットアルゴリズムを方策として使用）
        # 注意: アルゴリズムは現在の固定状態で行動選択
        micro_batch_data = dataset.obtain_batch_bandit_feedback(
            num_data=batch_size, 
            policy_algorithm=algorithm
        )
        
        # マイクロバッチ内の各データポイントで学習・評価
        for i in range(micro_batch_data.num_data):
            context = micro_batch_data.context_features[i]
            selected_actions = micro_batch_data.selected_action_vectors[i]
            rewards = micro_batch_data.observed_reward_vectors[i]
            available_actions_mask = micro_batch_data.available_action_mask[i]
            available_actions = np.where(available_actions_mask == 1)[0]
            
            # 最適報酬を計算（regret用）
            optimal_actions = []
            q_values = micro_batch_data.base_q_function[i, available_actions]
            sorted_indices = np.argsort(q_values)[::-1]
            for j in range(min(K, len(available_actions))):
                optimal_actions.append(available_actions[sorted_indices[j]])
            
            optimal_reward = sum([micro_batch_data.base_q_function[i, a] for a in optimal_actions])
            current_reward = sum(rewards)
            instant_regret = optimal_reward - current_reward
            
            instant_regrets.append(instant_regret)
            cumulative_regret += instant_regret
            
            # アルゴリズムを更新（リアルタイム学習）
            algorithm.update(
                context=context,
                selected_actions=selected_actions.tolist(),
                rewards=rewards.tolist()
            )
        
        # 進捗表示
        if (trial + batch_size) % 100 == 0 or (trial + batch_size) >= num_trials:
            print(f"  Trial {min(trial + batch_size, num_trials)}: "
                  f"累積Regret = {cumulative_regret:.3f}, "
                  f"平均瞬時Regret = {np.mean(instant_regrets[-batch_size:]):.3f}")

    # 5. 結果の表示
    print(f"\n=== 結果 ===")
    print(f"最終累積Regret: {cumulative_regret:.3f}")
    print(f"平均瞬時Regret: {np.mean(instant_regrets):.3f}")

    # 期間別の性能分析
    first_period_regret = np.mean(instant_regrets[0:200])
    second_period_regret = np.mean(instant_regrets[200:400])  
    third_period_regret = np.mean(instant_regrets[400:600])
    
    print(f"第1期間 (0-199) 平均Regret: {first_period_regret:.3f}")
    print(f"第2期間 (200-399) 平均Regret: {second_period_regret:.3f}")
    print(f"第3期間 (400-599) 平均Regret: {third_period_regret:.3f}")

    # 6. グラフ描画（オプション）
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(instant_regrets)
        plt.title('瞬時Regret')
        plt.xlabel('Trial')
        plt.ylabel('Instant Regret')
        plt.axvline(x=200, color='red', linestyle='--', alpha=0.7, label='Action Change')
        plt.axvline(x=400, color='red', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        cumulative_regrets = np.cumsum(instant_regrets)
        plt.plot(cumulative_regrets)
        plt.title('累積Regret')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Regret')
        plt.axvline(x=200, color='red', linestyle='--', alpha=0.7, label='Action Change')
        plt.axvline(x=400, color='red', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('microbatch_bandit_results.png', dpi=150, bbox_inches='tight')
        print(f"\nグラフを 'microbatch_bandit_results.png' に保存しました")
        
    except ImportError:
        print("\nmatplotlibが見つからないため、グラフは表示されません")

    print("\n=== マイクロバッチ学習完了 ===")


if __name__ == "__main__":
    main()