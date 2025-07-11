import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

scoring = {
    "ABORT to WIN": 3,
    "LOSS to WIN": {
        True: 2.5,  # additional criterion fulfilled (LOSS with low closeness or WIN in 4)
        False: 2
    },
    "faster": {
        True: 2,  # additional criterion fulfilled (high main score difference)
        False: 1
    },
    "ABORT to LOSS": 1,
    "closer": 0.5,
    "less close": -0.5,
    "LOSS to ABORT": -1,
    "slower": {
        False: -1,
        True: -2  # additional criterion fulfilled (high main score difference)
    },
    "WIN to LOSS": {
        False: -2,
        True: -2.5  # additional criterion fulfilled (WIN in 4 or LOSS with low closeness)
    },
    "WIN to ABORT": -3
}

# labels for impact score scale
yticklabels = [
    "WIN to ABORT",
    "WIN to LOSS *",
    "WIN to LOSS | slower *",
    "LOSS to ABORT | slower",
    "less close",
    "neutral",
    "closer",
    "ABORT to LOSS | faster",
    "LOSS to WIN | faster *",
    "LOSS to WIN *",
    "ABORT to WIN"
]

df_dir = os.path.join("..", "dataframes")
IG_by_guess_file = os.path.join(df_dir, "IG_by_guess.csv")
IG = pd.read_csv(IG_by_guess_file, index_col="episode")
targets = IG.target.to_list()
# print(IG.mean(numeric_only=True))


def merge_score_files(model_dir, game="wordle", expected_count=30):
    game_dir = os.path.join(model_dir, game)
    input_files = list(Path(game_dir).rglob("scores.json"))
    assert len(input_files) == expected_count, len(input_files)
    files_by_ep = {}
    for sc_file in input_files:
        parts = sc_file.parts
        exp_idx = parts.index(game) + 1
        offset = 0 if parts[exp_idx].startswith("0_high") else 10
        ep_id = offset + int(parts[exp_idx + 1].split('_')[1])
        assert ep_id not in files_by_ep
        files_by_ep[ep_id] = sc_file
    scores_dicts = []
    for ep_id in sorted(files_by_ep):
        with open(files_by_ep[ep_id], 'r', encoding="utf-8") as f_in:
            scores_dicts.append({"e": ep_id, **json.load(f_in)})
    # write result
    output_file = os.path.join(game_dir, "scores_by_episode.json")
    with open(output_file, 'w', encoding="utf-8") as f_out:
        json.dump(scores_dicts, f_out)


def build_df_from_scores(scores_by_episode):
    records = []  # to collect row data
    for ep_data in scores_by_episode:
        ep_id = ep_data["e"]
        episode_scores = ep_data["episode scores"]
        if episode_scores["Success"]:
            outcome = 1
        elif ep_data["episode scores"]["Lose"]:
            outcome = 0
        else:
            assert episode_scores["Aborted"]
            outcome = -1
        main_score = episode_scores["Main Score"]
        initial_closeness = ep_data["turn scores"]["1"]["closeness score"]
        # print(ep_data["e"], initial_closeness)
        if outcome == 0:
            assert "6" in ep_data["turn scores"]
            lost_with_closeness = ep_data["turn scores"]["6"]["closeness score"]
            # print(lost_with_closeness)
        else:
            lost_with_closeness = None
        records.append((ep_id, outcome, main_score, initial_closeness, lost_with_closeness))
    df = pd.DataFrame.from_records(
        records,
        index="episode",
        columns=[
            "episode",
            "outcome",
            "main_score",
            "initial_closeness",
            "lost_with_closeness",
        ],
    )
    return df


def get_df_from_csv(df_dir, prompt, model, take_guesses_from=None):
    csv_file = os.path.join(df_dir, f"{prompt}_{model}.csv")
    df = pd.read_csv(csv_file, index_col="episode")
    if take_guesses_from is not None and "opening_guess" not in df.columns:
        data_for_comp_file = os.path.join(
            take_guesses_from, f"{prompt}_{model}", "wordle",
            "data_for_computation.json"
        )
        df = add_opening_guess_to_df(df, data_for_comp_file)
        print(f"Updating {csv_file}")
        df.to_csv(csv_file)
    df["main_score"] = df["main_score"].fillna(value=0)
    return df


def add_opening_guess_to_df(df, data_for_comp_file, colname="opening_guess"):
    with open(data_for_comp_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    # add first turns' guesses, in correct order (as defined by targets)
    opening_guesses = [None] * len(targets)
    for ep_data in data:
        assert ep_data["target_word"] in targets
        target_idx = targets.index(ep_data["target_word"])
        opening_guesses[target_idx] = ep_data["turns_guess_feedback"][0][0]
    df[colname] = opening_guesses
    return df


def impact_on_episode(cot, b):
    if cot.outcome == b.outcome:
        if cot.outcome == 1:
            if cot.main_score > b.main_score:
                return ("positive", "faster")
            if cot.main_score == b.main_score:
                return ("neutral", "similar success")
            return ("negative", "slower")
        if cot.outcome == 0:
            if cot.lost_with_closeness > b.lost_with_closeness:
                return ("positive", "closer")
            if cot.lost_with_closeness == b.lost_with_closeness:
                return ("neutral", "similar loss")
            return ("negative", "less close")
        assert cot.outcome == -1
        return ("neutral", "both aborted")
    if cot.outcome < b.outcome:
        if cot.outcome == 0:
            assert b.outcome == 1
            return ("negative", "WIN to LOSS")
        assert cot.outcome == -1
        # played to aborted
        if b.outcome == 0:
            return ("negative", "LOSS to ABORT")
        assert b.outcome == 1
        return ("negative", "WIN to ABORT")
    assert cot.outcome > b.outcome
    if b.outcome == 0:
        assert cot.outcome == 1
        return ("positive", "LOSS to WIN")
    assert b.outcome == -1
    # aborted to played
    if cot.outcome == 0:
        return ("positive", "ABORT to LOSS")
    assert cot.outcome == 1
    return ("positive", "ABORT to WIN")


def impact_counters(impact_results):
    main_results_status = Counter()  # pos./neg./neutral impact on final status
    main_results_all = Counter()  # pos./neg./neutral impact in general
    finegrained_results = Counter()
    for category, detailed in impact_results:
        if " to " in detailed:
            assert category != "neutral"
            main_results_status[category] += 1
        else:
            main_results_status["neutral"] += 1
        main_results_all[category] += 1
        finegrained_results[detailed] += 1
    return (main_results_status, main_results_all, finegrained_results)


def better_was_easier(better, worse, method="initial_closeness"):
    # Is better played associated with higher initial closeness?
    if "opening_guess" in better.index:
        if better.opening_guess == worse.opening_guess:
            # print(better.opening_guess)
            return False
    if method == "initial_closeness":
        if better.initial_closeness > worse.initial_closeness:
            return True
        return False
    if method == "entropy":
        if "episode" in better.index:
            ep_id = better.episode
            assert "episode" in worse.index and ep_id == worse.episode
        else:
            ep_id = better.name
            assert ep_id == worse.name
        if IG.at[ep_id, better.opening_guess] > IG.at[ep_id, worse.opening_guess]:
            return True
        return False


def impact_score(cot, b):
    category, detailed = impact_on_episode(cot, b)
    if category == "neutral":
        return 0
    if isinstance(scoring[detailed], (int, float)):
        return scoring[detailed]
    if detailed == "LOSS to WIN":
        extra_fulfilled = b.lost_with_closeness < 15 or cot.main_score >= 50
    elif detailed in {"faster", "slower"}:
        extra_fulfilled = abs(cot.main_score - b.main_score) >= 30
    elif detailed == "WIN to LOSS":
        extra_fulfilled = b.main_score >= 50 or cot.lost_with_closeness < 15
    return scoring[detailed][extra_fulfilled]


if __name__ == "__main__":
    # Note: uses the directory df_dir defined in line 50
    models = ["gemma-3-27b", "gpt-4o-2024-08-06", "llama-3.3-70b"]
    variants = ["0", "1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c"]
    dfs_cot = []  # CoT dataframes (to concat later)
    dfs_lm = []  # dataframes (to concat later) used for lmplots
    main_score_gain = {}  # to collect data for overview 1: clemscore gain/loss
    impact_scores = {}  # to collect data for overview 2a: mean impact scores
    impact_scores_played = {}  # to collect data for overview 2b
    for model in models:
        b_df = get_df_from_csv(df_dir, "baseline", model)
        b_main_score_sum = b_df.main_score.sum()
        main_score_gain[model] = {}
        impact_scores[model] = {}
        impact_scores_played[model] = {}
        for prompt in variants:
            # compare CoT results to baseline (= without CoT)
            df = get_df_from_csv(df_dir, prompt, model)
            assert df.shape[0] == 30 == b_df.shape[0]
            # number of successful episodes (not sum because of -1 for aborted)
            # success = len(df[df["outcome"] == 1])
            # print(f"# {prompt}, {model}")
            # print(f"successful episodes: {success}")
            df["impact"] = df.apply(
                lambda row: impact_on_episode(row, b_df.loc[row.name]), axis=1
            )
            df["impact_score"] = df.apply(
                lambda row: impact_score(row, b_df.loc[row.name]), axis=1
            )
            impact_scores[model][prompt] = df.impact_score.mean()
            df_played = df[(df.outcome != -1) & (b_df.outcome != -1)]
            impact_scores_played[model][prompt] = df_played.impact_score.mean()

            percent = ((100 / b_main_score_sum) * df.main_score.sum()) - 100
            main_score_gain[model][prompt] = percent

            df = df.reset_index()
            df["model"] = model
            df["prompt"] = prompt
            dfs_cot.append(df)
            dfs_lm.append(df[(df.initial_closeness < 25) & (df.outcome != -1)])
        # add baseline data
        b_df = b_df.reset_index()
        b_df["model"] = model
        b_df["prompt"] = "baseline"
        dfs_lm.append(b_df[(b_df.initial_closeness < 25) & (b_df.outcome != -1)])

    output_dir = "output"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # write overviews to csv files
    overview_data = {
        "clemscore_gain": main_score_gain,
        "impact_scores": impact_scores,
        "impact_scores_played": impact_scores_played,
    }
    for descr, data in overview_data.items():
        overview_df = pd.DataFrame(data)
        overview_df["mean"] = overview_df.mean(axis=1)
        overview_df.loc["mean"] = overview_df.mean()
        overview_df = overview_df.round(3)
        # overview_df["beneficial_for"] = overview_df.apply(
            # lambda row: tuple(m for m in models if row[m] > 0), axis=1
        # )
        overview_df.to_csv(
            os.path.join(output_dir, f"overview-{descr}.csv")
        )
    cot_df = pd.concat(dfs_cot, ignore_index=True)
    cot_df["variant"] = cot_df.prompt.str[0]
    cot_df["subvariant"] = cot_df.prompt.str[-1]
    agg_impact_summary = cot_df.groupby(["model", "prompt"]).agg(
        {"impact": lambda col: impact_counters(col)[2].most_common(5)}
    )
    agg_impact_summary.to_csv(
        os.path.join(output_dir, "most-common-impacts.csv")
    )
    # create boxplots for impact score results
    colors = ["lightskyblue", "mediumslateblue", "paleturquoise"]
    med_colors = ["#42657A", "#2B2554", "#4F6B6B"]
    yticks = sum(
        [[s for s in v.values()] if type(v) is dict else [v] for v in scoring.values()],
        [0]
    )
    yticks = sorted(set(yticks))
    assert len(yticks) == len(yticklabels), (len(yticks), len(yticklabels))
    plot_by = ["variant", "subvariant"]
    for col_name in plot_by:
        fig, axes = plt.subplots(ncols=3, sharey=True)
        for i, m in enumerate(models):
            if col_name == "subvariant":  # exclude zero-shot variant
                sub_df = cot_df[(cot_df.model == m) & (cot_df.prompt != "0")]
            else:
                sub_df = cot_df[cot_df.model == m]
            a = sub_df.boxplot(
                column="impact_score",
                grid=False, by=col_name, ax=axes[i], patch_artist=True,
                boxprops={"facecolor": colors[i]},
                medianprops={"color": med_colors[i], "linewidth": 1.5},
                flierprops={"markerfacecolor": colors[i]}
            )
            a.set_title(m)
            a.set_xlabel("Prompt")
        # fig.suptitle("Impact Score by Model and Prompt Variant")
        fig.suptitle("")
        # fig.supylabel("Impact Score")
        fig.supylabel("")
        plt.yticks(yticks, yticklabels)
        save_as = os.path.join(output_dir, f"Impact-Score-by-{col_name}.pdf")
        plt.savefig(save_as, bbox_inches="tight")

    # Is there a (positive) correlation between the initial closeness/IG
    # and performance, measured by the main score?
    full_df_lm = pd.concat(dfs_lm, ignore_index=True)
    full_df_lm["initial_IG"] = full_df_lm.apply(
        lambda row: IG.at[row.episode, row.opening_guess], axis=1
    )
    # print(full_df_lm)
    # colors = ["lightskyblue", "mediumslateblue", "paleturquoise"]
    colors = ["lightskyblue", "mediumslateblue", "#4F6B6B"][::-1]
    ms_ylabel = "Main Score"
    ms_yticks = [0, 20, 30, 50, 100]  # all possible main scores
    # initial_closeness, without legend (for usage as left subfigure)
    sns.lmplot(
        x="initial_closeness", y="main_score", data=full_df_lm, fit_reg=True,
        hue="model", palette=colors, hue_order=models[::-1], legend=False
    )
    plt.xlabel("Initial Closeness")
    plt.xticks([0, 5, 10, 15, 20])
    plt.ylabel(ms_ylabel)
    plt.yticks(ms_yticks)
    save_as = os.path.join(output_dir, "Main-Score-vs-IC.pdf")
    plt.savefig(save_as, bbox_inches="tight")
    # initial_IG, with legend
    sns.lmplot(
        x="initial_IG", y="main_score", data=full_df_lm, fit_reg=True,
        hue="model", palette=colors, hue_order=models[::-1]
    )
    plt.xlabel("Initial Information Gain")
    plt.ylabel(ms_ylabel)
    plt.yticks(ms_yticks)
    save_as = os.path.join(output_dir, "Main-Score-vs-IIG.pdf")
    plt.savefig(save_as, bbox_inches="tight")
