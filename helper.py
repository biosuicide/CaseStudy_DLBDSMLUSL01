import textwrap
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from collections import OrderedDict
from math import pi

from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate

from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from matplotlib.projections.polar import PolarAxes
from typing import cast
from docx import Document



cb = sns.color_palette("colorblind")

def removing_rare_answers(
    col_threshold_dict: dict,
    data: pd.DataFrame
):
    for key, value in col_threshold_dict.items():
        shares = data[key].value_counts(normalize=True)
        rare = shares[shares < value].index

        data[key] = data[key].where(
            ~data[key].isin(rare),
            "Other" 
        )
        print("="*40)
        print(f"after removing rare answers in {key}:")
        print("="*40)
        print(data[key].value_counts())


def gender_map():
    male_terms = {
        "male","m","man",
        "cis male","cis man","male (cis)",
        "male.","sex is male", "dude",
        "i'm a man why didn't you make this a drop down question. you should of asked sex? and i would of answered yes please. seriously how much text can this take?",
        "malr","mail","m|","cisdude"
    }
    
    female_terms = {
        "female","f","woman",
        "cis female","cis woman","female/women",
        "female assigned at birth", "fm","fem",
        "female (props for making this a freeform field, though)",
        "cis-woman", "female-bodied; no feelings about gender"
    }

    d = {t: "male" for t in male_terms}
    d.update({t: "female" for t in female_terms})

    return d

def plot_circle_with_table(
    column: str,
    data: pd.DataFrame,
    title: str ="",
    savefile: bool =False,
    savefile_name:str ="unnamed_figure.jpeg"
):
    count = (
        data[column]
        .value_counts()
        .reindex(data[column].unique(), fill_value=0)
        .sort_values(ascending=False)
    )
    pct = (count / count.sum() * 100).round(1)

    fig, (ax_pie, ax_tbl) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [2.2, 1]}
    )

    colors = plt.get_cmap("Blues")(np.linspace(0.7, 0.2, len(count)))

    wedges, _ = ax_pie.pie(
        x=count.values,
        colors=colors,
        startangle=180,
        labels=None,
        wedgeprops={"linewidth": 1, "edgecolor": "white"}
    )

    ax_pie.axis("equal")
    ax_pie.set_xlim(-1.5, 1.5)
    ax_pie.set_ylim(-1.3, 1.3)

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey", lw=0.8)
    arrow_props = dict(arrowstyle="-", color="lightgrey", lw=1)

    for wedge, label in zip(wedges, count.index):
        angle = (wedge.theta2 + wedge.theta1) / 2
        angle_rad = np.deg2rad(angle)
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)
        x_text = 1.3 if x >= 0 else -1.3
        y_text = 1.2 * y
        ax_pie.annotate(
            f"{label}",
            xy=(x, y),
            xytext=(x_text, y_text),
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=10,
            bbox=bbox_props,
            arrowprops=dict(arrow_props, connectionstyle=f"angle,angleA=0,angleB={angle}")
        )

    # --- TABLE ---
    ax_tbl.axis("off")
    table_df = pd.DataFrame({"Working in": count.index, "%": pct.values})

    tbl = ax_tbl.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        colWidths=[0.78, 0.22],
        loc="center",
        cellLoc="left"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.4)

    # Header style
    for c in range(table_df.shape[1]):
        hdr = tbl[(0, c)]
        hdr.set_text_props(weight="bold")
        hdr.set_edgecolor("lightgrey")

    # Helper: choose black/white text by background luminance
    def rel_luminance(rgba):
        r, g, b, _ = rgba
        return 0.2126*r + 0.7152*g + 0.0722*b

    # Data style
    n_rows, n_cols = table_df.shape
    for r in range(1, n_rows + 1):
        bg = colors[r-1]
        txt = "black" if rel_luminance(bg) > 0.5 else "white"
        for c in range(n_cols):
            cell = tbl[(r, c)]
            cell.set_facecolor(bg)
            cell.set_edgecolor("lightgrey")
            cell.get_text().set_color(txt)
            if c == 1:  # right-align the % column
                cell.get_text().set_ha("right")

    fig.suptitle(title, fontsize=18, verticalalignment="center")
    fig.tight_layout()
    if savefile:
        fig.savefig(savefile_name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def scale_translate(
    geo_series, 
    xfact=1.0, 
    yfact=1.0, 
    xoff=0.0, 
    yoff=0.0
):
    geo_series = geo_series.apply(
        lambda g: shp_scale(
            g, 
            xfact=xfact, 
            yfact=yfact, 
            origin="center"
        )
    )
    geo_series = geo_series.apply(lambda g: shp_translate(g, xoff=xoff, yoff=yoff))
    return geo_series


def do_ohe_and_tsne(df, components=2):
    tsne = TSNE(n_components=components, random_state=42)
    ohe = OneHotEncoder(
        drop=None,
        sparse_output=False,
        dtype="float64",
        handle_unknown="ignore"
    )

    minmax = MinMaxScaler()
    categorical_cols = df.select_dtypes(include=["string", "object", "category"]).columns.tolist()
    numeric_cols = [c for c in df.columns if c not in categorical_cols]

    transformer = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_cols),
            ("num", minmax, numeric_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    ohe_cleaned_data = transformer.fit_transform(df)
    cleaned_data_tsne = tsne.fit_transform(ohe_cleaned_data)
    print(f"OHE data shape: {ohe_cleaned_data.shape}")
    return cleaned_data_tsne


def plot_tsne_by(col, df, tsne,  s=40, alpha=0.8):
    tsne_df = pd.DataFrame(tsne, columns=["tsne0", "tsne1"]).reset_index(drop=True)
    plot_df = pd.concat([tsne_df, df.reset_index(drop=True)], axis=1)

    unique_vals = plot_df[col].dropna().unique()

    # Start with empty dict
    palette = {}

    # Always fix Yes/No if present
    if "Yes" in unique_vals:
        palette["Yes"] = cb[0]
    if "No" in unique_vals:
        palette["No"] = cb[1]

    # Assign remaining categories in order, skipping reserved indices
    reserved = {0, 1}
    color_idx = 0
    for val in sorted(unique_vals):
        if val not in palette:
            # find the next color not reserved
            while color_idx in reserved:
                color_idx += 1
            palette[val] = cb[color_idx]
            color_idx += 1

    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(
        data=plot_df,
        x="tsne0", 
        y="tsne1",
        hue=col,
        palette=palette, 
        s=s, 
        alpha=alpha, 
        edgecolor=None
    )

    wrapped = "\n".join(textwrap.wrap(col, width=40))
    ax.set_title(f"{wrapped}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="lower right"
    )

    plt.tight_layout()
    plt.show()


def plot_multiple_tsne_by(
        col, 
        df, 
        tsne, 
        alt_title=[], 
        s=10, 
        alpha=0.8, 
        ax=None, 
        show_legend=True, 
        title_wrap=40
):
    if col not in df.columns:
        return ax  # nothing to draw

    # ensure matching length
    m = min(len(tsne), len(df))
    tsne_df = pd.DataFrame(tsne[:m], columns=["tsne0", "tsne1"]).reset_index(drop=True)
    plot_df = pd.concat([tsne_df, df[col].reset_index(drop=True).iloc[:m]], axis=1)

    unique_vals = plot_df[col].dropna().unique()
    # if hue is all-NaN or empty, bail
    if plot_df[col].dropna().empty:
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    palette = {}

    # Always fix Yes/No if present
    if "Yes" in unique_vals:
        palette["Yes"] = cb[0]   # blue

    if "No" in unique_vals:
        palette["No"] = cb[1]   # orange

    # Assign remaining categories in order, skipping reserved indices
    reserved = {0, 1}
    color_idx = 0
    for val in sorted(unique_vals):
        if val not in palette:
            # find the next color not reserved
            while color_idx in reserved:
                color_idx += 1
            palette[val] = cb[color_idx]
            color_idx += 1

    sns.scatterplot(
        data=plot_df, x="tsne0", y="tsne1",
        hue=col, palette=palette,
        s=s, alpha=alpha, edgecolor=None, ax=ax
    )

    # tidy legend and title
    if show_legend:
        ax.legend(
            loc="lower right"
        )
    else:
        leg = ax.get_legend()
        if leg: leg.remove()

    if alt_title != []:
        ax.set_title("\n".join(textwrap.wrap(str(alt_title), width=title_wrap)))
    
    else:
        ax.set_title("\n".join(textwrap.wrap(str(col), width=title_wrap)))
    
    return ax


def plot_tsne_grid(
    cols, 
    df, 
    tsne, 
    alt_title=[], 
    nrows=4, 
    ncols=4, 
    figsize=(8.27,4.5)
):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    all_hl = []

    for ax, col, alt_title in zip(axes_flat, cols, alt_title):
        plot_multiple_tsne_by(col, df, tsne, alt_title, ax=ax, show_legend=False, s=3, alpha=0.7)
        h, l = ax.get_legend_handles_labels()
        if h and l:
            all_hl.extend(zip(h, l))

    # remove unused axes
    for ax in axes_flat[len(cols):]:
        ax.axis("off")

    # remove y-label and ticks for all except the first
    for i, ax in enumerate(axes_flat[:len(cols)]):
        if i != 0:
            ax.set_yticks([])
            ax.set_ylabel("")

    # single legend outside
    if all_hl:
        # keep first handle per label
        uniq = OrderedDict()
        for h, l in all_hl:
            if l and l != "_nolegend_" and l not in uniq:
                uniq[l] = h

        # zip them together
        priority = {"Yes": 0, "No": 1}
        items = sorted(uniq.items(), key=lambda kv: (priority.get(kv[0], 2), kv[0]))

        # sort labels, so Yes/No are alwyas in the same place
        labels_sorted, handles_sorted = zip(*[(k, uniq[k]) for k, _ in items])

        fig.legend(
            handles_sorted, labels_sorted,
            loc="lower center",
            ncol=min(6, len(labels_sorted)),
            frameon=False,
            fontsize="x-large",
            markerscale=3
        )

    # leave room for bottom legend
    plt.tight_layout(rect=(0, 0.05, 1, 1))  
    return fig, axes


def print_confusion_matrix(
    y_pred,
    y_test
):
    cm = confusion_matrix(y_test, y_pred)
    print("="*40)
    print("Confusion Matrix:")
    print("="*40)
    print(cm)


def get_feature_importance(model, x, y):
    scoring = [
        "f1",
        "accuracy"
    ]

    permutation_result = permutation_importance(
        model, 
        x, 
        y,
        n_repeats=30,
        random_state=42,
        scoring=scoring
    )
    importances = pd.DataFrame({
        "feature": x.columns,
        "accuracy: importance_mean": permutation_result["accuracy"]["importances_mean"],
        "accuracy: importance_std": permutation_result["accuracy"]["importances_std"],
        "f1: importance_mean": permutation_result["f1"]["importances_mean"],
        "f1: importance_std": permutation_result["f1"]["importances_std"]
    }).sort_values(
        by="accuracy: importance_mean",
        ascending=False
    )

    return importances


def show_sequential_feature_selection(
    model, 
    direction,
    transformer,
    X_train,
    y_train,
    X_test,
    y_test,
):

    sfs = SequentialFeatureSelector(
        estimator=model,
        n_features_to_select=5,   # choose how many you want
        direction=direction,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    sfs_pipeline = Pipeline([
        ("transformer", transformer),  # your ColumnTransformer from above
        ("sfs", sfs),
        ("clf", clone(model)) 
    ])

    # --- fit and evaluate ---
    sfs_pipeline.fit(X_train, y_train)
    y_pred = sfs_pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("="*40)
    print("="*40)
    print("Sequential Feature Selection Metrics")
    print("="*40)
    print("="*40)
    print("Confusion Matrix:")
    print("="*40)
    print(cm)

    print("="*40)
    print("Classification Report:")
    print("="*40)
    print(classification_report(y_test, y_pred, digits=3))

    # --- selected features (in encoded space) ---
    encoded_feature_names = sfs_pipeline.named_steps["transformer"].get_feature_names_out()
    selected_mask = sfs_pipeline.named_steps["sfs"].get_support()
    selected_features = encoded_feature_names[selected_mask]

    print(f"Selected features ({direction} SFS):")
    for name in selected_features:
        print("-", name)


def plot_k_distance(pca_space, k):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(pca_space)
    dists, _ = nn.kneighbors(pca_space)
    kth = np.sort(dists[:, -1])  # distance to the k-th neighbor
    plt.figure(figsize=(8, 4))
    plt.plot(kth)
    plt.ylabel(f"Distance to {k}-th neighbor")
    plt.xlabel("Points sorted by distance")
    plt.title(f"k-distance plot (k={k})")
    plt.tight_layout()
    plt.show()
    return kth


def try_dbscan_grid(Z, eps_list, min_samples_list):
    rows = []
    for eps in eps_list:
        for ms in min_samples_list:
            labels = DBSCAN(eps=eps, min_samples=ms, metric="euclidean").fit(Z).labels_
            n_noise = np.sum(labels == -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # silhouette only valid with >= 2 clusters and not all noise
            sil = np.nan
            if n_clusters >= 2 and n_noise < len(labels):
                try:
                    sil = silhouette_score(Z, labels, metric="euclidean")
                except Exception:
                    sil = np.nan
            rows.append({"eps": eps, "min_samples": ms,
                         "clusters": n_clusters, "noise_frac": n_noise/len(labels),
                         "silhouette": sil})
    return pd.DataFrame(rows).sort_values(
        by=["clusters", "silhouette"], ascending=[False, False]
    )



def _feature_scores(
    result_df: pd.DataFrame,
    cluster_ids,
    global_means: pd.Series | None
):
    sub = result_df.loc[:, cluster_ids]
    if global_means is not None:
        # align index to be safe
        gm = global_means.reindex(result_df.index)
        diffs = (sub.subtract(gm, axis=0)).abs() # type: ignore
        score = diffs.max(axis=1)
    else:
        score = sub.max(axis=1) - sub.min(axis=1)
    return score


def _sorted_features(
    result_df: pd.DataFrame,
    feature_names: list[str],
    cluster_ids,
    global_means: pd.Series | None,
    top_n: int | None
):
    """Return feature_names sorted by score (desc), optionally truncated to top_n."""
    scores = _feature_scores(result_df, cluster_ids, global_means)
    feats = pd.Index(feature_names).intersection(result_df.index)
    ordered = feats.to_series().sort_values(key=lambda s: scores.loc[s], ascending=False)
    if top_n is not None:
        ordered = ordered.head(top_n)
    return list(ordered.values)


def top_features_per_cluster(
    df_with_labels, 
    cluster_col="cluster", 
    top_n=5
):
    # Separate features from cluster labels
    X = df_with_labels.drop(columns=[cluster_col])
    labels = df_with_labels[cluster_col]

    # Global proportions (mean of each binary feature across all samples)
    global_means = X.mean()

    results = {}

    # Loop over clusters
    for cluster_id in labels.unique():
        cluster_data = X[labels == cluster_id]
        cluster_means = cluster_data.mean()

        # Compare cluster proportions vs global proportions
        diffs = (cluster_means - global_means).abs().sort_values(ascending=False)
    
        # Pick top_n features
        top_feats = pd.DataFrame({
            "cluster_mean": cluster_means[diffs.index],
            "global_mean": global_means[diffs.index],
            "abs_diff": diffs
        }).head(top_n)

        results[cluster_id] = top_feats

    return results


def plot_cluster_feature_diffs(
    result_df, 
    feature_names, 
    cluster_ids
):
    subset = result_df.loc[feature_names, cluster_ids]
    ax = subset.plot(kind="barh", figsize=(20, len(feature_names)*2))
    plt.xlabel("Proportion (0–1)")
    plt.title("Cluster feature profiles")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()


def _split_str_fixed(text: str, n_splits: int, width: int = 25) -> str:
    """
    Wrap text into at most `n_splits + 1` lines.
    Example: n_splits=2 → max 3 lines total.
    """
    parts = textwrap.wrap(str(text), width=width)
    if len(parts) <= n_splits + 1:
        return "\n".join(parts)
    # merge leftovers into the last allowed line
    merged = parts[:n_splits] + [" ".join(parts[n_splits:])]
    return "\n".join(merged)


def _split_at_last_underscore(label: str) -> tuple[str, str]:
    """
    Split a label into (left, right) at the LAST underscore.
    Returns (whole_string, "") if no underscore found.
    """
    if "_" not in label:
        return label, ""
    left, right = label.rsplit("_", 1)
    return left, right


def plot_heatmap(
    result_df: pd.DataFrame,
    feature_names,
    cluster_ids,
    *,
    global_means: pd.Series,
    top_n: int,
    q_width: int = 25,
    q_splits: int = 1,
):
    ordered_features = _sorted_features(
        result_df, feature_names, cluster_ids, global_means, top_n
    )
    subset = result_df.loc[ordered_features, cluster_ids].copy()

    # Format row labels: split into question + answer
    new_index = []
    for lbl in subset.index:
        question, answer = _split_at_last_underscore(str(lbl))
        wrapped_q = _split_str_fixed(question, n_splits=q_splits, width=q_width)
        if answer:
            new_index.append(wrapped_q + "\n" + answer)
        else:
            new_index.append(wrapped_q)
    subset.index = pd.Index(new_index)

    # Optional: wrap cluster labels a bit
    x_labels = [_split_str_fixed(str(c), n_splits=1, width=10) for c in cluster_ids]

    # Scale figure size
    fig_width = len(cluster_ids) * 1.2 + 4
    fig_height = max(3.0, len(ordered_features) * 0.9 + 2)

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        subset,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        linewidths=0.5,
        linecolor="gray",
        vmin=0.0,
        vmax=1.0,
        xticklabels=x_labels,
    )
    ax.set_title("Heatmap: cluster proportions per feature")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def analyze_cluster_differences(
    cluster_profiles: pd.DataFrame,
    dbscan_transformer,
    output: bool = False,
    noise_col: int = -1,
    filter_threshold: float = 0.5,
    top_n: int = 10
) -> pd.DataFrame:
    clusters = [c for c in cluster_profiles.columns if c != noise_col]

    interesting_cluster = cluster_profiles.copy()
    ohe_feature_names = dbscan_transformer.get_feature_names_out()
    cluster_profiles.index = pd.Index(ohe_feature_names)

    # Differences to noise cluster
    for c in clusters:
        interesting_cluster[f"cluster_{c}_diff_noise"] = abs(cluster_profiles[c] - cluster_profiles[noise_col])

    # Pairwise differences between clusters
    for c1, c2 in itertools.combinations(clusters, 2):
        colname = f"cluster_{c1}_diff_{c2}"
        interesting_cluster[colname] = abs(cluster_profiles[c1] - cluster_profiles[c2])

    # Filter features with large differences
    diff_columns = [col for col in interesting_cluster.columns if "diff" in str(col)]
    filtered = interesting_cluster.loc[(interesting_cluster[diff_columns] >= filter_threshold).any(axis=1)]

    # Sort by maximum observed difference
    filtered["max_diff"] = filtered[diff_columns].max(axis=1)
    filtered_sorted = filtered.sort_values(by="max_diff", ascending=False)

    # Display top N features
    if output:
        for idx, row in filtered_sorted.head(top_n).iterrows():
            print(f"\n--- Feature: {idx} ---")
            for col in diff_columns:
                if row[col] >= filter_threshold:
                    print(f"{col}: {row[col]:.2f}")

    return filtered_sorted



def plot_radar_with_noise(df, counts, title, savefile = False, savefile_name= ""):
    """Plot a cleaner radar chart, show n in legend."""
    categories = df.columns.tolist()
    num_features = len(categories)

    angles = [i / float(num_features) * 2 * pi for i in range(num_features)]
    angles += angles[:1]  # close loop

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"polar": True})
    ax = cast(PolarAxes, ax)

    # Style: remove bold circular border, soften grid
    ax.spines["polar"].set_visible(False)
    ax.grid(True, linewidth=0.6, alpha=0.75)
    ax.set_yticklabels([]) 

    colors = sns.color_palette("colorblind", n_colors=len(df))

    for (cluster_name, row), color in zip(df.iterrows(), colors):
        values = row.tolist() + row.tolist()[:1]
        label_with_n = f"{cluster_name} (n={counts.get(cluster_name, '?')})"
        ax.plot(angles, values, linewidth=1.6, label=label_with_n, color=color)
        ax.fill(angles, values, alpha=0.18, linewidth=0, color=color)

    # Add a straight reference line on the right side (angle = 0)
    angle_shift = pi / 6  # 15 degrees
    ax.plot([angle_shift, angle_shift], [0, 1], color="grey", lw=0.8, ls="--", alpha=0.8)
    # ax.plot([0, 0], [0, 1], color="grey", lw=0.8, ls="--", alpha=0.8)
    for r in [0, 0.5, 1]:
        ax.text(angle_shift+0.05, r, f"{r:.1f}", color="grey", fontsize=8, va="center")

    # Angles and labels
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], categories, fontsize=9)

    # Title and legend
    ax.set_title(title, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False)
    
    plt.tight_layout()
    if savefile:
        fig.savefig(savefile_name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)

def plot_cluster_barplot(df_radar, cluster_counts, title="", savefile=False, savefile_name=""):
    """Plot a grouped horizontal barplot of cluster feature means (0–1 scale)."""
    # Reset index for seaborn (cluster column)
    df_long = df_radar.reset_index().melt(
        id_vars="index", var_name="Feature", value_name="Mean value"
    )
    df_long.rename(columns={"index": "Cluster"}, inplace=True)

    # Create readable cluster labels with counts
    df_long["Cluster"] = df_long["Cluster"].map(
        lambda c: f"{c} (n={cluster_counts.get(c, '?')})"
    )

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    # Horizontal grouped barplot
    sns.barplot(
        data=df_long,
        y="Feature",
        x="Mean value",
        hue="Cluster",
        orient="h",
        palette="colorblind"
    )

    plt.xlabel("Mean response (0–1 scale)", fontsize=11)
    plt.ylabel("")
    plt.title(title, fontsize=12, pad=12)

    # Make layout tight and readable
    plt.legend(
        title="Cluster",
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
        frameon=False
    )
    plt.xlim(0, 1)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile_name, dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()
    
def create_word_counts(
    col_to_analyze: str,
    ref_col: str,
    mask_name_dict: dict,
    df: pd.DataFrame, 
    n_gram_range: tuple = (3,4),
    min_occurances: int = 3,
    export: bool = False
):
    # custom stop word list, as we need to include e.g. "not"
    stop_words = [
        "is", "if", "it", "is", "to", "in", "the", "of", "and",
        "as", "are", "only", "my", "do", "had", "what", "how",
        "for", "eg", "120", "100", "20", "30", "at",
        "then", "yes", "that", "when", "after", "or", "an", "on",
        "would", "might", "me", "am", "was"
    ]
    
    # analyze = df[col_to_analyze]
    mask_dict = {
        mask_name_dict["Yes"]: df[ref_col].eq("Yes"),
        mask_name_dict["No"]: df[ref_col].eq("No"),
        mask_name_dict["Maybe"]: df[ref_col].eq("Maybe"),
    }

    count_vectorizer = CountVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words=stop_words,
        ngram_range=n_gram_range,
        min_df=min_occurances
    )

    tfidf_vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words=stop_words,
        ngram_range=n_gram_range,
        min_df=min_occurances,
        norm="l2",
        smooth_idf=True,
        sublinear_tf=True
    )

    dataframes = []

    for name, mask in mask_dict.items():
        answer = df.loc[mask, col_to_analyze]
        tfidf_weights = tfidf_vectorizer.fit_transform(answer)
        weight_sums = np.asarray(tfidf_weights.sum(axis=0)).ravel() # type:ignore

        count_matrix = count_vectorizer.fit_transform(answer)
        count_sums = np.asarray(count_matrix.sum(axis=0)).ravel() # type:ignore

        n_gram_freq = pd.DataFrame({
            "ngram": tfidf_vectorizer.get_feature_names_out(),
            "count": count_sums,
            "tfidf_sum": weight_sums.round(2)
        })

        n_gram_freq["tfidf_per_occurrence"] = (
            n_gram_freq["tfidf_sum"] / n_gram_freq["count"]
        ).round(2)

        sorted_top_20 = n_gram_freq.head(20).sort_values(by="count", ascending=False)

        if not export:
            print("="*40)
            print(name)
            print("="*40)
            print(sorted_top_20)

        if export:
            dataframes.append(sorted_top_20)

    if export:
        return dataframes
    
    else:
        return []
    
def create_table_in_word(df, name):
    doc = Document()
    table = doc.add_table(rows=1, cols=len(df.columns))
    for i, col in enumerate(df.columns):
        table.rows[0].cells[i].text = str(col)
    for row in df.itertuples(index=False):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.save(f"{name}.docx")