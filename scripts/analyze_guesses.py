import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

with open("recognized_words_english.json", "r", encoding="utf-8") as f:
    wordlist = set(json.load(f))


def main(game_dir, evaluate_explanations=False):
    """Creates a pandas.DataFrame with results from guess analyses.

    Conducts analyses using analyze_episode_guesses() for all episodes
    included in the file "data_for_computation.json" in game_dir.
    Each episode data is expected to be represented by a dictionary
    that has a key "turns_guess_feedback".
    If no such file is present, it gets written by gathering
    data_for_computation from all interactions.json files within
    game_dir (before calling analyze_episode_guesses()).

    Args:
      game_dir: The path (str) to the directory from which to take
        data_for_computation.json or interactions.json files.
      evaluate_explanations (bool): Defaults to False; if True,
        the data_for_computation files must contain the key
        "guess_explanation" for each epsiode. The guess explanations
        are passed to eval_structured_representation().

    Returns:
      A pandas.DataFrame, comprising metadata, e.g. the epsiode's
        target word, two measures of information gain as well as
        the total and individual counts of 10 types of strategic
        shortcomings of which "red_immediate", "red_distant",
        "yellow_immediate" and "yellow_distant" refer to repetitions,
        whereas the remaining types refer to omissions.
        Dimensionality: one row per turn, 19 columns,
        20 columns if evaluate_explanations
      An integer providing the count of analyzed episodes.
      A list of target words skipped due to empty turns_guess_feedback.
      A float representing the average number of guess repetitions per
        episode (based on turns_guess_feedback)
    """
    game_dir = Path(game_dir)
    v1_6 = "v1.6" in game_dir.parts
    data_for_comp_file = os.path.join(game_dir, "data_for_computation.json")
    # gather data_for_computation from all interactions files in game_dir
    # or work with data_for_comp_file if it already exists
    if not os.path.isfile(data_for_comp_file):
        input_files = list(game_dir.rglob("*interactions.json"))
        write_data_for_computation(input_files, data_for_comp_file)
    # read newly|previously created data_for_comp_file
    with open(data_for_comp_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not data:
            raise ValueError(f"{data_for_comp_file} is lacking data")
    # save player of first episode in data (for later assert statement)
    player = data[0]["player_1"]
    # prepare dataframe
    cols = [
        "target", "turn", "guess", "feedback",
        "expected_information_gain", "actual_information_gain",
        "total_count", "repetitions", "red_immediate", "red_distant",
        "yellow_immediate", "yellow_distant", "omissions",
        "green_at_correct_immediate", "green_at_correct_distant",
        "green_complete_immediate", "green_complete_distant",
        "yellow_complete_immediate", "yellow_complete_distant",
    ]
    if evaluate_explanations:
        cols.append("structured_representation")
    data_for_df = {key: [] for key in cols}
    # analyze all episodes and update data_for_df accordingly
    episode_count = 0
    empty_tgf = []  # episodes (target words) with empty turns_guess_feedback
    guess_repetition_count = 0
    for episode_data in data:
        target = episode_data["target_word"]
        assert player == episode_data["player_1"], (
            f"Expected {player}, not {episode_data["player_1"]}"
        )
        # analyze all guesses (results consist of one tuple per guess/turn)
        turns_guess_feedback = episode_data["turns_guess_feedback"]
        if turns_guess_feedback == []:
            empty_tgf.append(target)
            continue
        if evaluate_explanations:
            results, _ = analyze_episode_guesses(
                turns_guess_feedback,
                target,
                explanations=episode_data["guess_explanation"],
                v1_6=v1_6
            )
            repetitions, omissions, eig, ig, sr_eval = zip(*results)
        else:
            results, _ = analyze_episode_guesses(
                turns_guess_feedback,
                target,
                v1_6=v1_6
            )
            repetitions, omissions, eig, ig = zip(*results)
        n = len(results)
        assert n == len(turns_guess_feedback)
        # append metadata
        data_for_df["target"].extend(n * [target])
        data_for_df["turn"].extend(range(1, n + 1))
        guesses, feedbacks = zip(*turns_guess_feedback)
        data_for_df["guess"].extend(guesses)
        data_for_df["feedback"].extend(feedbacks)
        # to enhance the interpretability of the respective col.mean() values
        # in the df that is to be returned, set eig and ig to np.nan for
        # aborted episodes or if there's a missing (e)ig value
        aborted = episode_data["game_final_status"] == "ABORT"
        missing = [(e is np.nan) ^ (a is np.nan) for e, a in zip(eig, ig)]
        if aborted or any(missing):
            eig = ig = (n * [np.nan])
        # append results
        data_for_df["expected_information_gain"].extend(eig)
        data_for_df["actual_information_gain"].extend(ig)
        assert repetitions[0] is np.nan
        assert omissions[0] is np.nan
        for key in cols[6:19]:
            data_for_df[key].append(np.nan)
        for i in range(1, n):
            assert set(repetitions[i]).isdisjoint(omissions[i])
            turn_results = repetitions[i] + omissions[i]
            repetitions_sum = repetitions[i].total()
            omissions_sum = sum(
                v for k, v in omissions[i].items() if not k.startswith("green_c")
            )  # complete o. of green letters are also o. at correct
            for key in cols[6:19]:
                if key == "total_count":
                    data_for_df[key].append(repetitions_sum + omissions_sum)
                elif key == "repetitions":
                    data_for_df[key].append(repetitions_sum)
                elif key == "omissions":
                    data_for_df[key].append(omissions_sum)
                else:
                    data_for_df[key].append(turn_results[key])
        if evaluate_explanations:
            data_for_df["structured_representation"].extend(sr_eval)
        guess_repetition_count += guess_repetitions(guesses)
        episode_count += 1
    df = pd.DataFrame(data_for_df)
    guess_repetition_average = guess_repetition_count / episode_count
    return df, episode_count, empty_tgf, guess_repetition_average


def write_data_for_computation(interactions_files, output_file):
    data = [get_data_for_computation(f_in) for f_in in interactions_files]
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out)


def get_data_for_computation(interactions_file):
    data = {}
    with open(interactions_file, "r", encoding="utf-8") as f:
        interactions = json.load(f)
    try:
        return interactions["turns"][-1][-1]["action"]["data_for_computation"]
    except KeyError:
        data["player_1"] = interactions["players"]["Player 1"]
        eval_data = interactions["Evaluation"]
        data["turns_guess_feedback"] = eval_data["guess"]
        data["target_word"] = eval_data["target_word"]
        data["target_word_difficulty"] = eval_data["target_word_difficulty"]
        outcome = interactions["turns"][-1][-1]["action"]["content"]
        data["game_final_status"] = outcome.removeprefix("game_result = ")
        data["guess_explanation"] = eval_data["guess_explanation"]
        return data


def analyze_episode_guesses(
        turns_guess_feedback, target_word, explanations=None, v1_6=False
):
    """
    Expected form of turns_guess_feedback:
    A list of lists; inner lists like:
    ["serum", "s<green> e<green> r<green> u<green> m<green>"]

    Covers strategic shortcomings in terms of letter repetitions and
    omissions, which are subcategorized by color and turn distance:
    - repeated red and, depending on position, repeated yellow letters
    - green letters not present at known correct positions
    - completely omitted green or yellow letters (counted twice in the
      rare case that a letter is known to be correct in two positions)
    - each label (key in Counters) comes with the suffix "_immediate"
      or "_distant" to indicate whether the last guess feedback was
      sufficient to be aware of the respective contsraint or not.

    Note:
      Instances of green letters that are missing at the correct
      position but have been included in the guess can be computed
      by subtracting the "green_complete_*" counts from the
      "green_at_correct_*" counts in the omission Counters.
    """
    # initialize list for gathering results per turn (index = turn - 1)
    results = []
    # initialize containers to incrementally collect provided feedback
    red = set()
    yellow = defaultdict(set)
    green = defaultdict(set)
    recent_red = set()
    recent_yellow = defaultdict(set)
    recent_green = defaultdict(set)
    # initialize set of possible words
    remaining = wordlist.copy()
    # iterate over turns_guess_feedback to go through each feedback string
    for turn, fb in enumerate(turns_guess_feedback, 1):
        guess = fb[0]
        # evaluate the variables in structured representation if explanations
        if turn > 1 and explanations is not None:
            expl = explanations[turn - 1]  # -1 since turn index started at 1
            sr_eval = eval_structured_representation(expl, green, yellow, red)
        # prepare Counters for strategic shortcomings;
        # keys representing the subtypes are added at their first occurence
        repetition = Counter()
        omission = Counter()
        # initialize / reset containers for information in current feedback
        current_red = defaultdict(set)  # for practical reasons also a dict
        current_yellow = defaultdict(set)
        current_green = defaultdict(set)
        # analyze the guess letter by letter & prepare for next guess analysis
        for position, letter_fb in enumerate(fb[1].split(), 1):
            letter = letter_fb[0]
            color = letter_fb[2:-1]
            # where appropriate, update repetition and omission
            if turn > 1:
                if letter in red:
                    assert color == "red"
                    if letter in recent_red:
                        repetition["red_immediate"] += 1
                    else:
                        repetition["red_distant"] += 1
                elif letter in yellow and position in yellow[letter]:
                    rate = True
                    if color == "green":
                        # once something is marked green, it should not
                        # be rated as a bad repetition in later turns
                        yellow[letter].remove(position)
                        if len(remaining) == 0:
                            # player has no valid choice other than
                            # reconsidering previously yellow letters
                            rate = False
                    if rate:
                        # new placement failure I
                        if letter in recent_yellow and position in recent_yellow[letter]:
                            repetition["yellow_immediate"] += 1
                        else:
                            repetition["yellow_distant"] += 1
                if position in set().union(*green.values()) and color != "green":
                    if letter in green and position in green[letter]:
                        # not an omission
                        assert color == "yellow"
                    elif position in set().union(*recent_green.values()):
                        omission["green_at_correct_immediate"] += 1
                    else:
                        omission["green_at_correct_distant"] += 1
            # save current information to be available during next guess analysis
            if color == "red":
                current_red[letter].add(position)
            elif color == "yellow":
                current_yellow[letter].add(position)
            elif color == "green":  # expected to be equivalent to 'else' here
                current_green[letter].add(position)
        # complete omission of a known letter (covers new placement failure II)
        # omitted known letters in green && yellow are counted as green here
        included_letters = set(guess)
        for omitted_green in set(green).difference(included_letters):
            for known_position in green[omitted_green]:
                if known_position in set().union(*recent_green.values()):
                    omission["green_complete_immediate"] += 1
                else:  # i.e., not in the recent feedback or marked yellow
                    omission["green_complete_distant"] += 1
        yellow_only = set(yellow).difference(green)
        for omitted_yellow in yellow_only.difference(included_letters):
            if omitted_yellow in recent_yellow:
                omission["yellow_complete_immediate"] += 1
            else:
                omission["yellow_complete_distant"] += 1
        # handle the special case of both red and non-red feedback on a letter
        if len(included_letters) < 5:  # since only possibly when duplicates
            for red_green in set(current_red).intersection(current_green):
                # make use of yellow category to achieve wanted behaviour
                if red_green in current_yellow or v1_6:  # add red position(s)
                    current_yellow[red_green].update(current_red[red_green])
                else:  # add all except the green positions
                    green_positions = green[red_green] | current_green[red_green]
                    current_yellow[red_green].update(
                        {1, 2, 3, 4, 5} - green_positions
                    )
                # either way, remove from red
                current_red.pop(red_green)
            for red_yellow in set(current_red).intersection(current_yellow):
                current_yellow[red_yellow].update(current_red[red_yellow])
                current_red.pop(red_yellow)
        # create (if 1st turn) / overwrite variables holding recent information
        recent_red = set(current_red)  # positions don't need to be remembered
        recent_yellow = current_yellow
        recent_green = current_green
        # update episode-level feedback containers
        red.update(recent_red)
        for yl in recent_yellow:
            if recent_yellow[yl].isdisjoint(green[yl]):
                if not green[yl]:
                    green.pop(yl)
                yellow[yl].update(recent_yellow[yl])
            else:
                yellow[yl].update(recent_yellow[yl] - green[yl])
        for gl in recent_green:
            green[gl].update(recent_green[gl])
        # calculate information gain and update remaining possible solutions
        eig = np.nan  # overwrite if applicable
        ig = np.nan  # overwrite if applicable
        if len(remaining) > 1 and target_word in remaining:
            # i.e., there is something to gain
            # and the given information did not seemingly rule out the target word
            eig = expected_information_gain(guess, remaining)
            H = math.log(len(remaining), 2)  # uncertainty before current guess
            remaining, _ = possible_words(remaining, red, yellow, green)
            if target_word in remaining:
                ig = H - math.log(len(remaining), 2)
        elif target_word not in remaining:
            remaining, _ = possible_words(wordlist.copy(), red, yellow, green)
        # save turn results
        if turn > 1:
            turn_results = (repetition, omission, eig, ig)
            if explanations is not None:
                turn_results += (sr_eval,)
        else:
            turn_results = (np.nan, np.nan, eig, ig)
            if explanations is not None:
                turn_results += (np.nan,)
        results.append(turn_results)
    assert red.isdisjoint(green)
    assert red.isdisjoint(yellow)
    return results, remaining


def add_hatching(ax, df, legend=True):
    m, n = df.shape
    patterns = ["", "//"] * int(n / 2)  # to hatch every other bar
    hatches = [p for p in patterns for i in range(m)]
    bars = ax.patches
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        bar.set_edgecolor("#424242")  # a dark grey
    if legend:
        ax.legend()


def possible_words(words, red, yellow, green):
    new_remaining = set()
    for w in words:
        if red.intersection(w):
            continue  # check next word
        green_fulfilled = True
        for letter, positions in green.items():
            for pos in positions:
                if w[pos - 1] != letter:
                    green_fulfilled = False
                    break
            if not green_fulfilled:
                break  # abort checking green constraints
        if not green_fulfilled:
            continue  # check next word
        yellow_fulfilled = True
        for letter, positions in yellow.items():
            if letter not in w:
                yellow_fulfilled = False
                break  # abort checking yellow constraints
            for pos in positions:
                if w[pos - 1] == letter:
                    yellow_fulfilled = False
                    break
            if not yellow_fulfilled:
                break  # abort checking yellow constraints
        if not yellow_fulfilled:
            continue  # check next word
        # implicit else case: word w fits all constraints
        new_remaining.add(w)
    n_eliminated = len(words) - len(new_remaining)
    percent_eliminated = (100 / len(words)) * n_eliminated
    return new_remaining, percent_eliminated


def expected_information_gain(guess, remaining):
    n = len(remaining)
    distribution = defaultdict(float)
    for possible_target in remaining:
        pattern = ""
        for i, letter in enumerate(guess):
            if letter not in possible_target:
                pattern += "r"
            elif possible_target[i] == letter:
                pattern += "g"
            else:
                pattern += _yellow_or_red(guess, possible_target, i, letter)
        distribution[pattern] += 1 / n
    eig = 0
    for _pattern, prob in distribution.items():
        eig += prob * math.log((1 / prob), 2)
    return eig


def _yellow_or_red(guess, target, i, letter):
    total_count = Counter(target)[letter]
    upcoming_green = sum(
        [a == b == letter for a, b in zip(guess[i + 1:], target[i + 1:])]
    )
    max_yellow = total_count - upcoming_green
    if Counter(guess[:i + 1])[letter] <= max_yellow:
        return "y"
    else:
        return "r"


def guess_repetitions(guesses):
    # almost the same as clembench "total guess repetitions" (in scores.json),
    # just not NaN for aborted episodes,
    # as long as there was at least one accepted guess
    if not 1 <= len(guesses) <= 6:
        raise ValueError(f"Expected one to six guesses, got {len(guesses)}")
    # if len(guesses) > len(set(guesses)):
        # print(guesses)
    return len(guesses) - len(set(guesses))


def invalid_words(game_dir, assert_n_episodes=None):
    interactions_files = list(Path(game_dir).rglob("*interactions.json"))
    if assert_n_episodes is not None:
        info = f"\nexpected {assert_n_episodes}, got {len(interactions_files)}"
        assert len(interactions_files) == assert_n_episodes, game_dir + info
    error_count = 0
    for file in interactions_files:
        with open(file, "r", encoding="utf-8") as f:
            interactions = json.load(f)
        for turn in interactions["turns"]:
            for ia in turn:
                if ia["from"] == "GM" == ia["to"]:
                    if "NOT_VALID_WORD_FOR_GAME" in ia["action"]["content"]:
                        error_count += 1
    return error_count


def eval_structured_representation(explanation, green, yellow, red):
    v_eval = {}
    variable_names = ["keep", "try_in_position", "avoid"]
    non_green_pos = {1, 2, 3, 4, 5}.difference(*green.values())
    gold = [
        green,
        {k: non_green_pos - v for k, v in yellow.items() if k not in green},
        red
    ]
    # note that for try_in_position it is an approximation of the true gold;
    # with targets containing duplicate letters, it is possible that one of the
    # correct positions has been indentified (added to green) and that it is
    # also revealed that there is an additional occurence for which the correct
    # placement is yet to be found, so that the letter should still be included
    # in try_in_position (under all other circumstances, however, it is correct
    # to not include letters in try_in_position when they are in green)
    expl_text = explanation.replace("\n    ", "").replace("\n}", "}")
    for v, v_gold in zip(variable_names, gold):
        pattern = re.compile(rf"{v}\s*=\s*[^\n]+", re.IGNORECASE)
        matches = pattern.findall(expl_text)
        if matches:
            try:
                v_model = eval(matches[-1].split("=")[1].split("#")[0].strip())
                if v_model == v_gold:
                    v_eval[v] = 1  # correct
                elif isinstance(v_model, dict):
                    v_model_ = {
                        let: {pos} if type(pos) is int else pos
                        for let, pos in v_model.items()
                    }
                    if v_model_ == v_gold:
                        v_eval[v] = 1  # also counts as correct
                    else:
                        v_eval[v] = 0  # incorrect, at best incomplete
                else:
                    v_eval[v] = 0  # incorrect, at best incomplete
            except SyntaxError:
                v_eval[v] = -1  # not evaluable
        elif not v_gold:
            v_eval[v] = 1  # also counts as correct
        else:
            v_eval[v] = -2  # code for no match for a non empty variable
    return v_eval


# util. functions
def get_long_dir_name(model_results_dir):
    return ("--").join([model_results_dir] * 2)


def extra_df_struct_repr(df, prompt_model):
    # create new dataframe to facilitate working with struct. repr. results
    include_cols = ["target", "total_count", "structured_representation"]
    new_df = df[include_cols][df.turn > 1]
    # 'unpack' dictionary
    for var in ["keep", "try_in_position", "avoid"]:
        new_df[var] = new_df.apply(
            lambda row: row.structured_representation[var], axis=1
        )
    new_df.drop(columns=["structured_representation"], inplace=True)
    new_df["total_count"] = new_df["total_count"].astype(int)
    new_df["model"] = prompt_model[3:]
    new_df["variant"] = prompt_model[0]
    new_df["subvariant"] = prompt_model[1]
    return new_df


if __name__ == "__main__":
    # Specify data/directories to be analyzed
    if len(sys.argv) in {2, 3}:
        benchmark_run = sys.argv[1]
        # allow a substring filter
        model_substr = sys.argv[2] if len(sys.argv) == 3 else None
    else:
        print(
            "Please specify `CoT` or, for a benchmark run directory,\n`v1.6` "
            "or `v2.0` as in `python analyze_guesses.py v2.0`.\n"
            "Note that v1.6 and v2.0 are expected to be sibling directories\n"
            "of the thesis Appendix (i.e., assumed to be relative paths from\n"
            "the shared parent directory).\n"
            "A second argument may be specified in order to use a substring "
            "filter,\nfor example `gemma` or (with CoT) `1a_`."
        )
        exit()
    if benchmark_run == "v1.6":
        benchmark_run_dir = os.path.join("..", "..", benchmark_run)
        models = [
            get_long_dir_name("claude-3-5-sonnet-20240620-t0.0"),
            get_long_dir_name("Meta-Llama-3.1-405B-Instruct-Turbo-t0.0"),
            get_long_dir_name("gpt-4o-2024-08-06-t0.0"),
            get_long_dir_name("o1-preview-2024-09-12-t0.0"),
        ]
        models_short = [
            "claude-3-5-20240620",
            "Llama-3.1-405B",
            "gpt-4o-2024-08-06",
            "o1-preview"
        ]
        prompt_prefix = False  # just the model name
    elif benchmark_run == "v2.0":
        benchmark_run_dir = os.path.join("..", "..", benchmark_run)
        models = [
            get_long_dir_name("gemma-3-27b-it-t0.0"),
            get_long_dir_name("gpt-4o-2024-08-06-t0.0"),
            get_long_dir_name("Llama-3.3-70B-Versatile-t0.0"),
            get_long_dir_name("claude-3-7-sonnet-20250219-t0.0"),
            get_long_dir_name("deepseek-r1-t0.0"),
            get_long_dir_name("o3-mini-2025-01-31-t0.0"),

        ]
        models_short = [
            "gemma-3-27b",
            "gpt-4o-2024-08-06",
            "Llama-3.3-70B-V",  # added for the thesis (V for Versatile)
            "claude-3-7",
            "deepseek-r1",
            "o3-mini"
        ]
        # models_short = [m.split("-t0.0")[0] for m in models]  # alternative
        prompt_prefix = False  # just the model name
    elif benchmark_run == "CoT":
        benchmark_run_dir = os.path.join("..", "results")
        models = sorted(
            [d for d in os.listdir(benchmark_run_dir)
             if os.path.isdir(os.path.join(benchmark_run_dir, d))]
        )
        models_short = models
        prompt_prefix = True  # model names have a prefix specifying the prompt
        baseline_data = {
            f"baseline_{m}": os.path.join("..", "baseline_data", f"baseline_{m}", "wordle")
            for m in ["gemma-3-27b", "gpt-4o-2024-08-06", "llama-3.3-70b"]
        }
    else:
        print(
            "Accepted first arguments are: v1.6, v2.0 and CoT.\n"
            "A second argument may be specified in order to use a substring "
            "filter,\nfor example `gemma` or (with CoT) `1a_`."
        )
        exit()
    assert len(models) == len(models_short)
    for i, abbr in enumerate(models_short):
        for part in abbr.split("-"):
            assert part in models[i]
    print(f"Taking data from {benchmark_run_dir} as input.")

    # PREPARATIONS
    game = "wordle"
    output_dir = "output"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    benchmark_run_abbr = benchmark_run.removesuffix(".0").replace(".", "-")
    if model_substr is not None:
        benchmark_run_abbr += "-" + model_substr
    excel = False  # set to True to get detailed data as Excel spreadsheet
    if excel:
        xlsx_file = os.path.join(
            output_dir, f"turn_analysis_{game}_{benchmark_run_abbr}.xlsx"
        )
        writer = pd.ExcelWriter(xlsx_file)
        # long column names to rename
        col = {
            "expected_information_gain": "EIG",
            "actual_information_gain": "IG",
        }
    cols_plot_rep = [
        "red_immediate", "red_distant", "yellow_immediate", "yellow_distant"
    ]
    cols_plot_om = [
        "green_at_correct_immediate", "green_at_correct_distant",
        "green_complete_immediate", "green_complete_distant",
        "yellow_complete_immediate", "yellow_complete_distant"
    ]
    cols_plot_ig = ["expected_information_gain", "actual_information_gain"]
    data_plot_rep = {key: [] for key in cols_plot_rep}
    data_plot_om = {key: [] for key in cols_plot_om}
    data_plot_ig = {key: [] for key in cols_plot_ig}

    analyzed_episodes_per_model = []  # e.g., to take maximum count for ylabel
    guess_repetitions_by_model = {}  # average per analyzed episode
    invalid_word_counts_by_model = {}
    struct_repr = []  # only used if benchmark_run == "CoT"
    total_count_means = []
    included_experiments = []

    # RUN ANALYSIS for each model, gather data for plots, opt. write Excel
    for model_dir, m_short in tqdm(zip(models, models_short), total=len(models)):
        if model_substr is not None and model_substr not in m_short:
            continue
        game_dir = os.path.join(benchmark_run_dir, model_dir, game)
        if benchmark_run == "CoT" and model_dir[0] in {"1", "3"}:
            main_results = main(game_dir, evaluate_explanations=True)
            df, episode_count, empty_tgf, guess_rep = main_results
            struct_repr.append(extra_df_struct_repr(df, m_short))
        else:
            df, episode_count, empty_tgf, guess_rep = main(game_dir)
        analyzed_episodes_per_model.append(episode_count)
        guess_repetitions_by_model[m_short] = guess_rep
        if empty_tgf:
            print(f"INFO regarding {m_short}:")
            print(
                f"Skipped episodes {empty_tgf} due to empty turns_guess_feedback."
            )
        for key in data_plot_rep:
            data_plot_rep[key].append(df[key].mean() * 10)  # * 10 --> 10 turns
        for key in data_plot_om:
            data_plot_om[key].append(df[key].mean() * 10)  # * 10 --> 10 turns
        for key in data_plot_ig:
            data_plot_ig[key].append(df[key].mean())
        total_count_means.append((df["total_count"].mean(), m_short))
        if excel:
            df.rename(columns=col, inplace=True)
            df.rename(columns=lambda a: a.replace("_", " "), inplace=True)
            df["EIG"] = df["EIG"].round(2)
            df["IG"] = df["IG"].round(2)
            sheet_name = m_short[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        # assert_n = 30 if benchmark_run == "CoT" else None
        assert_n = 30
        invalid_word_count = invalid_words(game_dir, assert_n_episodes=assert_n)
        invalid_word_counts_by_model[m_short] = invalid_word_count
        included_experiments.append(m_short)
    if excel:
        writer.close()
        print(f"Detailed results have been written to {xlsx_file}.")

    # Create plots and other overviews ########################################

    max_n = max(analyzed_episodes_per_model)
    ylabel_1 = f"Total Count in {max_n} Episodes"
    ylabel_2 = f"Average (included episodes: {max_n})"
    ylabel_3 = "Average Count per 10 Turns"
    y_label_ig = "Average Information Gain per Turn (in bits)"

    if benchmark_run == "CoT" and model_substr is None:
        # plot/summarize selected CoT results
        # A) structured representation
        df_struct_repr = pd.concat(struct_repr, ignore_index=True)
        df_struct_repr.rename(columns={"total_count": "rep_om_count"}, inplace=True)
        kta_cols = ["keep", "try_in_position", "avoid"]
        # create Counters for all categories (1 (gold), 0, -1, -2) by model
        counters_per_var_by_model = df_struct_repr.groupby(["model"]).apply(
            lambda c: [Counter(c[var]) for var in kta_cols],
            include_groups=False
        )
        counters_per_var_by_model.to_csv(
            os.path.join(output_dir, "counters-per-var-by-model.csv")
        )
        # take evaluable variables only (= exclude turn if categories -1 or -2)
        # now we can use sum and mean
        df_struct_repr = df_struct_repr[
            df_struct_repr.apply(lambda r: all(r[v] >= 0 for v in kta_cols), axis=1)
        ]
        df_struct_repr["n_gold"] = df_struct_repr[kta_cols].sum(axis=1)
        sr_by_model = df_struct_repr.groupby(["model"]).mean(numeric_only=True).round(2)
        sr_by_model.to_csv(
            os.path.join(output_dir, "structured-representation.csv")
        )
        # print(df_struct_repr.mean(numeric_only=True).round(2))
        # print(df_struct_repr[df_struct_repr.rep_om_count > 5].to_string())
        # print(df_struct_repr.groupby(["variant"]).mean(numeric_only=True).round(2))
        # print(df_struct_repr.groupby(["subvariant"]).mean(numeric_only=True).round(2))
        # print(df_struct_repr.groupby(["model", "subvariant"]).mean(numeric_only=True).round(2))
        model_order = ["gemma-3-27b", "gpt-4o-2024-08-06", "llama-3.3-70b"]
        colors = ["lightskyblue", "mediumslateblue", "#4F6B6B"]
        sns.lmplot(
            x="n_gold", y="rep_om_count", data=df_struct_repr, fit_reg=True,
            hue="model", palette=colors, hue_order=model_order
        )
        plt.xlabel("Number of variables equivalent to computed gold")
        plt.xticks([0, 1, 2, 3])
        plt.ylabel("Repetition/Omission error count")
        save_as = os.path.join(output_dir, "structrepr-vs-repomcount.pdf")
        plt.savefig(save_as, bbox_inches="tight")

        # B) guess repetitions: in prompt 0 vs. 1∪2 vs. 3
        assert prompt_prefix
        df_gr = pd.DataFrame.from_dict(
            guess_repetitions_by_model, orient="index", columns=["guess_repetitions"]
        )
        df_gr["model"] = df_gr.index
        df_gr["model"] = df_gr.model.apply(lambda x: x[x.index("_") + 1:])
        df_gr["prompt"] = df_gr.index.str[0]
        df_gr["prompt"] = df_gr.prompt.apply(lambda x: "1∪2" if x in {"1", "2"} else x)
        if len(set(analyzed_episodes_per_model)) == 1:
            df_gr.guess_repetitions *= analyzed_episodes_per_model[0]
            df_gr.guess_repetitions = df_gr.guess_repetitions.astype(int)
            ylabel_gr = ylabel_1
        else:
            ylabel_gr = ylabel_2
        fig, axes = plt.subplots(ncols=3, sharey=True)
        colors = ["lightskyblue", "mediumslateblue", "paleturquoise"]
        med_colors = ["#42657A", "#2B2554", "#4F6B6B"]
        for i, m in enumerate(df_gr["model"].unique()):
            sub_df = df_gr[df_gr.model == m]
            a = sub_df.boxplot(
                grid=False, by="prompt", ax=axes[i], patch_artist=True,
                boxprops={"facecolor": colors[i]},
                medianprops={"color": med_colors[i], "linewidth": 1.5},
                flierprops={"markerfacecolor": colors[i]}
            )
            a.set_title(m)
            a.set_xlabel("Prompt")
        fig.suptitle("Guess Repetitions by Model and Prompt Variant")
        fig.supylabel(ylabel_gr)
        save_as = os.path.join(output_dir, f"{benchmark_run_abbr}-guessrepetitions.pdf")
        plt.savefig(save_as, bbox_inches="tight")
        # C) invalid words (5 letters, but not Wordle words in the language)
        for model, game_dir in baseline_data.items():
            baseline_count = invalid_words(game_dir, assert_n_episodes=30)
            invalid_word_counts_by_model[model] = baseline_count
        df_inv = pd.DataFrame.from_dict(
            invalid_word_counts_by_model, orient="index", columns=["invalid_words"]
        )
        df_inv["model"] = df_inv.index
        df_inv["model"] = df_inv.model.apply(lambda x: x[x.index("_") + 1:])
        df_inv["prompt"] = df_inv.index
        df_inv["prompt"] = df_inv.prompt.apply(lambda x: "Baseline" if x[0] == "b" else "CoT")
        fig, axes = plt.subplots(ncols=3, sharey=True)
        colors = ["lightskyblue", "mediumslateblue", "paleturquoise"]
        med_colors = ["#42657A", "#2B2554", "#4F6B6B"]
        for i, m in enumerate(df_inv["model"].unique()):
            sub_df = df_inv[df_inv.model == m]
            a = sub_df.boxplot(
                grid=False, by="prompt", ax=axes[i], patch_artist=True,
                boxprops={"facecolor": colors[i]},
                medianprops={"color": med_colors[i], "linewidth": 1.5},
                flierprops={"markerfacecolor": colors[i]}
            )
            a.set_title(m)
            a.set_xlabel("Prompt")
        fig.suptitle("Invalid Words by Model, Baseline vs. CoT")  # NOT_VALID_WORD_FOR_GAME
        fig.supylabel("Total Count per Prompt Variant (30 Episodes each)")
        save_as = os.path.join(output_dir, f"{benchmark_run_abbr}-invalidwords.pdf")
        plt.savefig(save_as, bbox_inches="tight")

    if benchmark_run != "CoT":
        # provide guess repetition counts
        print("\n# Number of guess repetitions:")
        for (model, g), n in zip(guess_repetitions_by_model.items(), analyzed_episodes_per_model):
            print(f"{model}: {int(g * n)} in {n} episodes")

    if benchmark_run == "CoT" and model_substr is None:
        exit()

    # undesired letter repetitions and omissions, then information gain
    legend = True
    legend = False
    df_index = included_experiments
    if benchmark_run == "CoT" and len(set([e.split("_")[1] for e in included_experiments])) == 1:
        # overwrite data for plots to only include selected CoT results
        total_count_means.sort()
        # print(total_count_means)
        select = [
            total_count_means[0][1],  # overall least rep./om. errors
            total_count_means[1][1],  # second least
            total_count_means[-1][1]  # most
        ]
        print(select)
        select_indices = [included_experiments.index(e) for e in select]
        b, _, w = select_indices
        assert included_experiments[b] == select[0]
        assert included_experiments[w] == select[-1]
        data_plot_rep = {
            k: [v[i] for i in select_indices] for k, v in data_plot_rep.items()
        }
        data_plot_om = {
            k: [v[i] for i in select_indices] for k, v in data_plot_om.items()
        }
        data_plot_ig = {
            k: [v[i] for i in select_indices] for k, v in data_plot_ig.items()
        }
        df_index = select
        # add baseline data
        for model, game_dir in baseline_data.items():
            if model_substr is not None and model_substr not in model:
                continue
            df, _, empty_tgf, _ = main(game_dir)
            if empty_tgf:
                print(f"INFO regarding {model} (baseline data)")
                print(
                    f"Skipped episodes {empty_tgf} due to empty turns_guess_feedback."
                )
            for key in data_plot_rep:
                data_plot_rep[key].append(df[key].mean() * 10)
            for key in data_plot_om:
                data_plot_om[key].append(df[key].mean() * 10)
            for key in data_plot_ig:
                data_plot_ig[key].append(df[key].mean())
            df_index.append(model)

    # plot repetitions
    df_rep = pd.DataFrame(data_plot_rep)
    df_rep.index = df_index
    if benchmark_run == "CoT":
        df_rep = df_rep.reindex([df_index[-1]] + df_index[:-1])
    colormap = ListedColormap(["#e62e00", "#ff5c33", "#ffff00", "#ffffb3"])
    plot_rep = df_rep.plot(kind="bar", colormap=colormap, legend=legend)
    add_hatching(plot_rep, df_rep, legend=legend)
    if legend:
        plt.legend(fontsize="small", frameon=False)
    plot_rep.tick_params(axis="x", labelrotation=12)
    if benchmark_run == "CoT":
        plt.yticks([i / 10 for i in range(0, 85, 10)])
    plt.ylabel(ylabel_3)
    plt.title("Repetition of red and yellow (at excluded position) letters")
    save_as = os.path.join(output_dir, f"{benchmark_run_abbr}-repetitions.pdf")
    plt.savefig(save_as, bbox_inches="tight")

    # plot omissions
    df_om = pd.DataFrame(data_plot_om)
    df_om.index = df_index
    if benchmark_run == "CoT":
        df_om = df_om.reindex([df_index[-1]] + df_index[:-1])
    colormap = ListedColormap(
        ["#538D4E", "#80b77b", "#66ff66", "#b3ffb3", "#ffff00", "#ffffb3"]
    )
    plot_om = df_om.plot(kind="bar", colormap=colormap, legend=legend)
    add_hatching(plot_om, df_om, legend=legend)
    if legend:
        plt.legend(fontsize="small", frameon=False)
    plot_om.tick_params(axis="x", labelrotation=12)
    plt.yticks([i / 10 for i in range(0, 35, 5)])
    plt.ylabel(ylabel_3)
    plt.title("Omission of green and yellow letters")
    save_as = os.path.join(output_dir, f"{benchmark_run_abbr}-omissions.pdf")
    plt.savefig(save_as, bbox_inches="tight")
    # provide (print) a table version of plot_rep & plot_om
    rep_om_df = pd.concat([df_rep, df_om], axis=1)
    rep_om_df["sum"] = rep_om_df.sum(axis=1)
    print("\n# undesired letter repetitions/omissions (average per 10 turns):")
    print(rep_om_df.round(2))
    # print(rep_om_df.round(2).to_string())

    # information gain
    df_ig = pd.DataFrame(data_plot_ig)
    df_ig.index = df_index
    if benchmark_run == "CoT":
        df_ig = df_ig.reindex([df_index[-1]] + df_index[:-1])
    colormap = ListedColormap(["#66ff66", "#538D4E"])
    plot_ig = df_ig.plot(kind="bar", colormap=colormap, legend=legend)
    if legend:
        plt.legend(fontsize="small", frameon=False)
    plot_ig.tick_params(axis="x", labelrotation=12)
    # plt.yticks([i / 10 for i in range(0, 35, 5)])
    plt.ylabel(y_label_ig)
    plt.title("Information Gain")
    save_as = os.path.join(output_dir, f"{benchmark_run_abbr}-IG.pdf")
    plt.savefig(save_as, bbox_inches="tight")
