import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

OUTPUT_PATH = "data/raw/gameplay_data.csv"


def generate_fair_player(player_id, match_id):
    shots_fired = np.random.randint(50, 300)
    shots_hit = np.random.randint(10, min(shots_fired, 120))
    accuracy = shots_hit / shots_fired
    kills = np.random.randint(0, 25)
    deaths = np.random.randint(0, 20)
    assists = np.random.randint(0, 15)
    headshots = np.random.randint(0, min(shots_hit, 20))
    reaction_time = np.random.normal(250, 50)   # ms
    avg_speed = np.random.normal(5.5, 1.0)
    max_speed = avg_speed + np.random.uniform(1, 4)
    resources_collected = np.random.randint(50, 500)
    damage_done = np.random.randint(100, 5000)
    session_duration = np.random.randint(300, 3600)
    suspicious_file_change = 0

    return [
        player_id, match_id, session_duration, kills, deaths, assists,
        shots_fired, shots_hit, accuracy, headshots, reaction_time,
        avg_speed, max_speed, resources_collected, damage_done,
        suspicious_file_change, 0
    ]


def generate_cheater(player_id, match_id):
    shots_fired = np.random.randint(50, 250)
    shots_hit = np.random.randint(int(shots_fired * 0.8), shots_fired)
    accuracy = shots_hit / shots_fired
    kills = np.random.randint(20, 60)
    deaths = np.random.randint(0, 5)
    assists = np.random.randint(0, 10)
    headshots = np.random.randint(int(shots_hit * 0.5), shots_hit)
    reaction_time = np.random.normal(80, 20)    # unrealistically low
    avg_speed = np.random.normal(9.5, 1.5)
    max_speed = avg_speed + np.random.uniform(3, 8)
    resources_collected = np.random.randint(500, 1500)
    damage_done = np.random.randint(4000, 12000)
    session_duration = np.random.randint(300, 3600)
    suspicious_file_change = np.random.choice([0, 1], p=[0.3, 0.7])

    return [
        player_id, match_id, session_duration, kills, deaths, assists,
        shots_fired, shots_hit, accuracy, headshots, reaction_time,
        avg_speed, max_speed, resources_collected, damage_done,
        suspicious_file_change, 1
    ]


def generate_dataset(num_samples=5000, cheat_ratio=0.2):
    data = []
    cheat_count = int(num_samples * cheat_ratio)
    fair_count = num_samples - cheat_count

    columns = [
        "player_id", "match_id", "session_duration", "kills", "deaths",
        "assists", "shots_fired", "shots_hit", "accuracy", "headshots",
        "avg_reaction_time_ms", "avg_speed", "max_speed",
        "resources_collected", "damage_done",
        "suspicious_file_change", "label"
    ]

    for i in range(fair_count):
        data.append(generate_fair_player(player_id=i, match_id=np.random.randint(1000, 9999)))

    for i in range(fair_count, fair_count + cheat_count):
        data.append(generate_cheater(player_id=i, match_id=np.random.randint(1000, 9999)))

    random.shuffle(data)
    return pd.DataFrame(data, columns=columns)


def main():
    os.makedirs("data/raw", exist_ok=True)

    df = generate_dataset(num_samples=5000, cheat_ratio=0.2)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(df.head())
    print("\nLabel distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()