# Above only applies to the model, does not applies to the 10 long iterations, I run to get the stats
# Cuml
import cudf as cudf
import cuml
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cuml.linear_model import Lasso
from cupy import asnumpy

cp.cuda.Device().use()
from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
from cuml import LinearRegression
from cuml.linear_model import LinearRegression
import pandas as pd
from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
from cuml.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
from cuml.linear_model import LinearRegression
import matplotlib.patches as mpatches
from collections import defaultdict
# import rmm  # RAPIDS Memory Manager
# rmm.reinitialize(pool_allocator=True, initial_pool_size=19 * 1024 ** 3)  # Use up to 16 GB of GPU memory
import os
import seaborn as sns


def compare_three_datasets1_fx1(train1, train2, train3, var1, var2, var3, output_dir, var_save, num_bins):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # num_bins = 50
    bin_edges = np.linspace(0, 1, num_bins + 1)

    blue_binned, _ = np.histogram(train1["LinScore"], bins=bin_edges)
    red_binned, _ = np.histogram(train2["LinScore"], bins=bin_edges)
    green_binned, _ = np.histogram(train3["LinScore"], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    space = 1.5
    width = 0.2
    blue_centers = bin_centers - (width + space)
    red_centers = bin_centers
    green_centers = bin_centers + (width + space)

    axes[0].bar(blue_centers, blue_binned, width=width, label=var1, align='center', color='blue')
    axes[0].bar(red_centers, red_binned, width=width, label=var2, align='center', color='red')
    axes[0].bar(green_centers, green_binned, width=width, label=var3, align='center', color='green')
    axes[0].set_title('Global Data Distribution for Three Datasets')
    axes[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axes[0].set_xlabel('Three Datasets (Downsampled)')
    axes[0].set_ylabel('Counts')
    axes[0].legend()

    all_binned_data = np.array([blue_binned, red_binned, green_binned])
    bin_positions = np.arange(num_bins)
    width = 0.30
    space = 5
    axes[1].bar(bin_positions - width, all_binned_data[0], width=width, label=var1, color='blue')
    axes[1].bar(bin_positions, all_binned_data[1], width=width, label=var2, color='red')
    axes[1].bar(bin_positions + width, all_binned_data[2], width=width, label=var3, color='green')
    axes[1].set_title('Bins-wise Data Distribution for Three Datasets ')
    axes[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axes[1].set_xlabel('Bins')
    axes[0].set_ylabel('Counts')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{var_save}")
    plt.show()


def compare_three_datasets1_fx2(train1_df, train2_df, train3_df, bins, output_dir, var_save):
    plt.hist(
        train1_df["LinScore"], bins=bins, alpha=0.5,
        color="blue", label="train_df_BalA", edgecolor="black", linewidth=1.2, density=True
    )

    plt.hist(
        train2_df["LinScore"], bins=bins, alpha=0.5,
        color="red", label="train_df_BalB", edgecolor="black", linewidth=1.2, density=True
    )

    plt.hist(
        train3_df["LinScore"], bins=bins, alpha=0.5,
        color="green", label="train_df_BalC", edgecolor="black", linewidth=1.2, density=True
    )

    sns.kdeplot(train1_df["LinScore"], color="blue", lw=2, label="train_df_BalA (KDE)")
    sns.kdeplot(train2_df["LinScore"], color="red", lw=2, label="train_df_BalB (KDE)")
    sns.kdeplot(train3_df["LinScore"], color="green", lw=2, label="train_df_BalC (KDE)")

    plt.title('Density and KDE for Three Datasets')
    plt.xlabel('Lin Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{output_dir}/{var_save}")


def graph_multi_plots1(ax, real, pred, title, alpha=0.10, s=1, color='blue'):
    """
    Plots real vs predicted values in a given axis with a unified y-axis range.
    """
    ax.scatter(real, pred, alpha=alpha, s=s, color=color)
    ax.plot([min(real), max(real)], [min(real), max(real)], 'k--', lw=2)  # Reference line
    # ax.set_xlabel("Real Y")
    ax.set_ylabel("Predicted Y")
    ax.set_title(title)
    y_min = min(pred)
    y_max = max(pred)
    ax.set_ylim(y_min, y_max)


def graph_multi_plots2(x_data, y_pred_1, var1, y_pred_2, var2, y_pred_3, var3, output_dir, var_save, var_title):
    x_data = np.array(x_data)
    # need to accommodate for the NaiveModel
    if len(list(set(y_pred_1))) == 1:
        y_pred_1 = np.array(y_pred_1)
        y_pred_2 = np.array(y_pred_2)
        y_pred_3 = np.array(y_pred_3)

    # need to accommodate for the NaiveModel
    if len(list(set(y_pred_1))) == 1:
        fig, axs = plt.subplots(1, 3)

    # need to accommodate for the NaiveModel
    if len(list(set(y_pred_1))) == 1:
        graph_multi_plots1(axs[0], x_data, y_pred_1, "Single Value Prediction", alpha=0.5, s=10, color='blue')
        graph_multi_plots1(axs[1], x_data, y_pred_2, "Single Value Prediction", alpha=0.5, s=10, color='red')
        graph_multi_plots1(axs[2], x_data, y_pred_3, "Single Value Prediction", alpha=0.7, s=10, color='green')
        for ax in axs[:3]:
            ax.set_ylim(0, 1)
        legend_patches = [
            mpatches.Patch(color='blue', label=var1),
            mpatches.Patch(color='red', label=var2),
            mpatches.Patch(color='green', label=var3)]
        fig.legend(handles=legend_patches, fontsize=10, title="Datasets", bbox_to_anchor=(1.37, 0.90))

    plt.suptitle(f'{var_title}', y=0.98)

    # legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{var_save}", bbox_inches="tight")


def percentage_r2_and_mse(y_real, y_pred):
    dic_tmp = defaultdict(list)

    if len(y_real) != len(y_pred):
        raise ValueError("y_real and y_pred must have the same length.")

    list_answers = list(zip(y_real, y_pred))
    list_answers = list_answers[:]
    bins = np.linspace(0, 1, 101)
    bins_list_cleaned = [[round(float(bins[i]), 2), round(float(bins[i + 1]), 2)] for i in range(len(bins) - 1)]
    bins_list_cleaned.insert(0, [0])
    bins_list_cleaned.append([1])

    for unit_r_p in list_answers:
        for bin_unit in bins_list_cleaned:
            if type(bin_unit[0]) != int:  # Special case: Bin (0.99, 1)
                if bin_unit[0] == 0.99 and bin_unit[1] == 1:
                    if unit_r_p[0] > 0.99 and unit_r_p[0] < 1:
                        dic_tmp[(bin_unit[0], bin_unit[1])].append(unit_r_p)
                        break
                else:  # General case for all bins except (0.99, 1)
                    if bin_unit[0] < unit_r_p[0] <= bin_unit[1]:
                        dic_tmp[(bin_unit[0], bin_unit[1])].append(unit_r_p)
                        break  # Stop checking other bins
            else:  # Special bins for exact 0 and 1
                if unit_r_p[0] == 0:
                    dic_tmp[0].append(unit_r_p)
                    break  # Stop checking other bins
                elif unit_r_p[0] == 1:
                    dic_tmp[1].append(unit_r_p)
                    break

    dic_tmp_real = defaultdict(list)
    dic_tmp_pred = defaultdict(list)
    for each_key in dic_tmp.keys():
        for each_r_p in dic_tmp[each_key]:
            dic_tmp_real[each_key].append(each_r_p[0])
            dic_tmp_pred[each_key].append(each_r_p[1])

    count_dic = {}
    for x in dic_tmp.keys():
        count_dic[x] = len(dic_tmp[x])

    dic_return_MSE = {}
    for key in dic_tmp_real.keys():
        mse_value = mean_squared_error(dic_tmp_real[key], dic_tmp_pred[key])
        dic_return_MSE[key] = mse_value

    dic_return_r2 = {}
    for key in dic_tmp_real.keys():
        if len(dic_tmp_real[key]) > 2:
            r2value = r2_score(dic_tmp_real[key], dic_tmp_pred[key])
            dic_return_r2[key] = r2value
        else:
            pass

    # sort the dic_return dic
    list1 = [dic_return_MSE, dic_return_r2]
    dic1_return_mse, dic1_return_r2 = {}, {}
    # list2=[dic1_return_mse,dic1_return_r2,count1_dic]

    for index in range(0, len(list1)):
        tmp1 = {}
        for key, value in list1[index].items():
            if isinstance(key, tuple):
                averaged_key = sum(key) / len(key)
                tmp1[averaged_key] = value
            else:
                tmp1[key] = value
        if index == 0:
            dic1_return_mse = dict(sorted(tmp1.items()))
        if index == 1:
            dic1_return_r2 = dict(sorted(tmp1.items()))

    return dic1_return_mse, dic1_return_r2, count_dic


def graph_stats_across_bins(var1, label1, var2, label2, var3, label3, title1, y_label, fig_name):
    perct = list(np.arange(0, 1.01, 0.01))
    # pre_last = .9901
    # perct.insert(-1, pre_last)

    plt.figure(figsize=(10, 5))
    plt.plot(perct, var1, marker='.', linestyle='dotted', color='blue', label=label1)
    plt.plot(perct, var2, marker='.', linestyle='dotted', color='red', label=label2)
    plt.plot(perct, var3, marker='.', linestyle='dotted', color='green', label=label3)

    plt.title(title1)
    plt.xlabel('100 bins in the Labels Range: 0 to 1')
    plt.ylabel(f'{y_label} per Each Bin in Labels Range')
    plt.grid(True)
    plt.legend()
    plt.savefig(fig_name)
    plt.show()


def calculate_global_weighted_and_unweighted_mse_and_r2(real, pred, counts, var1):
    return_df, df_original = pd.DataFrame(), pd.DataFrame()

    mse_value = mean_squared_error(real.to_list(), pred)
    r2value = r2_score(real.to_list(), pred)
    count_new = {}
    for key in counts:
        count_new[key] = 1 / counts[key]
    real_v_weight = {}

    for real_v in real.to_list():
        for tuples in count_new.keys():
            if type(tuples) != int:  # Special case: Bin (0.99, 1)
                if tuples[0] == 0.99 and tuples[1] == 1:
                    if 0.99 < real_v < 1:  # Check real_v in range (0.99, 1), excluding 1
                        real_v_weight[real_v] = count_new[tuples]
                        break
                else:  # Default logic for non-integer bins
                    if tuples[0] < real_v <= tuples[1]:
                        real_v_weight[real_v] = count_new[tuples]
                        break  # Stop checking further bins
            else:  # Special cases for exact 0 and 1
                if real_v == 0:
                    real_v_weight[real_v] = count_new[tuples]
                    break
                elif real_v == 1:
                    real_v_weight[real_v] = count_new[tuples]
                    break

    df_weighted = pd.DataFrame.from_dict(real_v_weight, orient="index").reset_index()
    df_weighted.columns = ["Original Real Value", "Weight"]

    df_original["Original Real Value"] = real
    df_original["Predicted"] = pred
    df_original_with_weight = pd.merge(df_original, df_weighted, on="Original Real Value", how="inner")
    weighted_mean_sq_error = mean_squared_error(df_original_with_weight["Original Real Value"].to_list(),
                                                df_original_with_weight["Predicted"].to_list(),
                                                sample_weight=df_original_with_weight["Weight"].to_list())

    weighted_r2 = r2_score(df_original_with_weight["Original Real Value"].to_list(),
                           df_original_with_weight["Predicted"].to_list(),
                           sample_weight=df_original_with_weight["Weight"].to_list())

    return_df["Model"] = [var1]
    return_df["MSE"] = [mse_value]
    return_df["Weighted MSE"] = [weighted_mean_sq_error]
    return_df["R2"] = [r2value]
    return_df["Weighted_R2"] = [weighted_r2]
    # print(len(count_new), len(counts))
    return return_df
