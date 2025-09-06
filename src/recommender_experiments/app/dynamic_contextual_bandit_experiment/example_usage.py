"""動的候補プール用Contextual Banditの使用例."""

import numpy as np
from recommender_experiments.service.environment.dynamic_contextual_bandit_dataset import (
    DynamicContextualBanditDataset,
    ItemLifecycle,
)


def main():
    """動的候補プール用Contextual Banditの使用例."""

    # アイテムライフサイクルを定義
    # 例：ニュース記事のようなコンテンツを想定
    item_lifecycles = [
        ItemLifecycle(item_id=0, start_day=0, duration_days=5),  # 5日間有効な記事
        ItemLifecycle(item_id=1, start_day=0, duration_days=3),  # 3日間有効な記事
        ItemLifecycle(item_id=2, start_day=1, duration_days=4),  # 1日目から4日間有効
        ItemLifecycle(item_id=3, start_day=2, duration_days=3),  # 2日目から3日間有効
        ItemLifecycle(item_id=4, start_day=3, duration_days=2),  # 3日目から2日間有効
        ItemLifecycle(item_id=5, start_day=4, duration_days=2),  # 4日目から2日間有効
    ]

    # シミュレーション設定
    dataset = DynamicContextualBanditDataset(
        dim_context=8,  # コンテキスト特徴量の次元数（ユーザの属性など）
        max_num_actions=10,  # アイテムID空間のサイズ
        item_lifecycles=item_lifecycles,
        days=7,  # 7日間のシミュレーション
        daily_traffic=1000,  # 1日1000トライアル
        reward_noise=0.2,  # 報酬ノイズの標準偏差
        beta=2.0,  # ソフトマックス温度（高いほど探索的）
        random_state=42,
    )

    # シミュレーション実行
    print("シミュレーション実行中...")
    result = dataset.generate_simulation_data()

    # 結果の表示
    print(f"\n=== シミュレーション結果 ===")
    print(f"総日数: {result.days}")
    print(f"1日あたりのtraffic: {result.daily_traffic}")
    print(f"総trial数: {result.total_trials}")

    print(f"\n=== 日別詳細 ===")
    for i, daily_data in enumerate(result.daily_data):
        avg_reward = np.mean(daily_data.observed_rewards) if len(daily_data.observed_rewards) > 0 else 0
        print(
            f"Day {i}: 利用可能アイテム={sorted(daily_data.available_actions)}, "
            f"平均報酬={avg_reward:.3f}, "
            f"trials={len(daily_data.observed_rewards)}"
        )

    print(f"\n=== 累積報酬推移 ===")
    for i, cum_reward in enumerate(result.cumulative_rewards):
        daily_reward = cum_reward if i == 0 else cum_reward - result.cumulative_rewards[i - 1]
        print(f"Day {i}: 日別報酬={daily_reward:.3f}, 累積報酬={cum_reward:.3f}")

    # アイテム別の選択頻度分析
    print(f"\n=== アイテム別選択頻度分析 ===")
    all_selections = {}
    for daily_data in result.daily_data:
        for action in daily_data.selected_actions:
            all_selections[action] = all_selections.get(action, 0) + 1

    total_selections = sum(all_selections.values())
    for item_id in sorted(all_selections.keys()):
        frequency = all_selections[item_id]
        percentage = frequency / total_selections * 100
        print(f"アイテム {item_id}: {frequency}回選択 ({percentage:.1f}%)")

    # アイテムライフサイクルと選択の関係を可視化
    print(f"\n=== アイテムライフサイクル vs 選択状況 ===")
    for lifecycle in item_lifecycles:
        selections_during_life = 0
        for day in range(lifecycle.start_day, lifecycle.end_day):
            if day < len(result.daily_data):
                day_selections = result.daily_data[day].selected_actions
                selections_during_life += np.sum(day_selections == lifecycle.item_id)

        total_available_slots = (lifecycle.end_day - lifecycle.start_day) * result.daily_traffic
        selection_rate = selections_during_life / total_available_slots * 100 if total_available_slots > 0 else 0

        print(
            f"アイテム {lifecycle.item_id}: {lifecycle.start_day}日目-{lifecycle.end_day - 1}日目 "
            f"(期間{lifecycle.duration_days}日), 選択率={selection_rate:.2f}%"
        )


if __name__ == "__main__":
    main()
