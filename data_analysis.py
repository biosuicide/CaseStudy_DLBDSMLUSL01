#%% libraries
import pandas as pd
import numpy as np
import importlib
import json
import helper  
importlib.reload(helper)

import country_converter as coco
import geopandas as gpd
import wget
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn import set_config
set_config(transform_output="pandas")
from bs4 import BeautifulSoup


# global plotting defaults
cb = sns.color_palette("colorblind")

#loading initial data
raw_data = pd.read_csv("mental-heath-in-tech-2016_20161114.csv")

#%% initial description of the dataset
# general matrix
print("got matrix with:")
print(f"{raw_data.shape[0]} rows")
print(f"and {raw_data.shape[1]} columns")
print("="*40)

# datatypes
print(f"Column Names:")
print("="*40)
print(raw_data.columns)
print("="*40)
print(f"Datatypes:")
print("="*40)
print(raw_data.dtypes.value_counts())
print("="*40)

# Missing values summary
missing_vals = raw_data.isnull().sum()
missing_percent = (missing_vals / len(raw_data)) * 100

print("Complete columns")
print("="*40)
for col, count in missing_vals.items():
    if count == 0:
        print(col)

missing_summary = pd.DataFrame({
    'Column': missing_vals.index,
    'Missing_Count': missing_vals.values,
    'Missing_Percentage': missing_percent.values
}).sort_values('Missing_Percentage', ascending=False)

print("="*40)
print("Columns with missing values:")
print("="*40)
print(missing_summary[missing_summary['Missing_Count'] > 0].head(44))
print("="*40)

#%% inspecting columns
print("="*40)
print("Initial statistics on numerical columns")
print("="*40)

numerical_columns = raw_data.select_dtypes(include=["int64", "float64"])

print(numerical_columns.describe())

print("="*40)
print("Inspection on object columns")
print("="*40)

object_columns = raw_data.select_dtypes(include="object")

for column in object_columns.columns:
    unique_values = object_columns[column].value_counts()
    unique_df = unique_values.reset_index()
    unique_df.columns = ["answer", "count"]
    print("="*40)
    print(f"{column}")
    print("="*40)
    print(f"unique values: {unique_values.count()}")
    print("Top 5:")
    print(unique_df.head(5))

#%% create data_dictionary
# reading json file and create dataframe
with open("osmi-survey-2016_1479139902.json", "r", encoding="utf-8") as json_data:
    survey = json.load(json_data)
    questions = survey["questions"]

questions_df = pd.DataFrame(questions)

# remove html tags in the questions
questions_df["clean_question"] = questions_df["question"].apply(
    lambda q: BeautifulSoup(q, "html.parser").get_text() if isinstance(q, str) else q
)

# mapping group names from IDs
group_questions = questions_df[questions_df["id"].str.startswith("group_")] 
group_dict = {
    row["id"]: row["clean_question"]
    for _, row in group_questions.iterrows()
}

# creating the data dictionary
data_dictionary = questions_df[["clean_question", "group", "id"]].copy()
data_dictionary = data_dictionary.iloc[2:].reset_index(drop=True)

# replacing group IDs with the names
data_dictionary["group"] = data_dictionary["group"].replace(group_dict)

# removing all rows, where it only descibes the group (no questions)
data_dictionary = data_dictionary[~data_dictionary["id"].str.startswith("group_")]

# mapping question types
type_patterns = {
    "multiple choice": r"^(list_|dropdown_)",
    "yes/no": r"^yesno_",
    "free text": r"^(textarea_|textfield_)",
    "number": r"^number_"
}
data_dictionary["question type"] = "other"

for q_type, pattern in type_patterns.items():
    data_dictionary.loc[data_dictionary["id"].str.match(pattern), "question type"] = q_type

# mapping information from raw data to data_dictionary
answered_percent = {
    col: 100 * raw_data[col].notnull().mean()
    for col in raw_data.columns
}

count_unique_values = {
    col: len(raw_data[col].unique())
    for col in raw_data.columns
}

data_type_mapping ={
    col: raw_data[col].dtype
    for col in raw_data.columns
}

data_dictionary["% answered"] = data_dictionary["clean_question"].map(answered_percent)
data_dictionary["unique values"] = data_dictionary["clean_question"].map(count_unique_values)
data_dictionary["raw_type"] = data_dictionary["clean_question"].map(data_type_mapping)

# removing duplicates
data_dictionary = data_dictionary.drop_duplicates(
    "id"
).reset_index(drop=True)
data_dictionary = data_dictionary.drop("id", axis=1)
data_dictionary.to_excel("raw_data_dictionary.xlsx", index=False)
# some duplicated values, but this is ok for now

# %% change binary mapping to categorical label
cleaned_data = raw_data.copy()
binary_mapping = {
    0: "No",
    1: "Yes"
}

# binary columns to "Yes" and "No", so it´s displayed properly in plots
print("="*40)
print(f"transforming binary columns")
print("="*40)
for col in cleaned_data:
    if cleaned_data[col].dtype != "object":
        cleaned_data[col] = cleaned_data[col].replace(
            to_replace=binary_mapping.keys(), #type: ignore
            value=binary_mapping.values()   #type: ignore
        )

#%% expanding multiple-answer questions
print("="*40)
print("Expand questions with many unique values to individual columns")
print(f"Before expansion: {cleaned_data.shape}")
print("="*40)

questions = {
    "If yes, what condition(s) have you been diagnosed with?": "own_diagnosis_yes",
    "If maybe, what condition(s) do you believe you have?": "own_diagnosis_maybe",
    "If so, what condition(s) were you diagnosed with?": "professional_diagnosis",
    "Which of the following best describes your work position?": "work_position"
}

new_df = pd.DataFrame()
for question, new_col_name in questions.items():
    if question not in cleaned_data.columns:
        continue 

    qcol = cleaned_data[question].astype("object")

    # normalize to lowercase, split at "|" and expand to columns 
    parts = (
        qcol
        .str.lower()
        .str.split("|", regex=False, expand=True)
    )

    # remove everything in parentheses, so we keep the disorder types only
    parts = parts.replace(r"\([^)]*\)", "", regex=True)
    parts = parts.apply(lambda s: s.str.strip())
    parts.columns = [f"{new_col_name}_{i}" for i in range(parts.shape[1])]

    # replace empty strings and "None" with NaN
    parts = parts.replace({"": pd.NA, "None": pd.NA, "none": pd.NA})
    new_df = pd.concat([new_df, parts], axis=1)

# concat the new df with the cleaned df and drop original columns
cleaned_data = pd.concat([cleaned_data, new_df], axis=1)
cleaned_data = cleaned_data.drop(
    labels=questions.keys(), # type:ignore
    axis=1
)

print(f"After expansion: {cleaned_data.shape}")
print("="*40)
#%% check which columns have a lot of NaN

# 0.7 seems reasonable, with 0.5 only one additional column is dropped
threshold = 0.7
total_values = cleaned_data.shape[0]

print("="*40)
print(f"Before cleaning: {cleaned_data.shape}")
print(f"dropping columns with more than {threshold*100}% missing Values")
print("="*40)

for col in cleaned_data.columns:
    percent_nan = cleaned_data[col].isna().sum()/total_values
    if  percent_nan >= threshold:
        # print(f"{col} has: \n {percent_nan:.1%} missing values")
        # print("dropping column")
        cleaned_data = cleaned_data.drop(
            labels=col,
            axis=1
        )
print(f"After cleaning: {cleaned_data.shape}")
print("="*40)

#%% 
# only one column left for professional diagnosis and own diagnosis
prof_diagnosis_type_col = "professional_diagnosis_0"
own_diagnosis_type_col = "own_diagnosis_yes_0"

# 2 cols for work position left
work_position_col_0 = "work_position_0"
work_position_col_1 = "work_position_1"

# group all rare answers to "other"
# 1% seems reasonable, 5% yields only 2 diagnosis types
threshold = 0.01  

shares = cleaned_data[prof_diagnosis_type_col].value_counts(normalize=True)
rare = shares[shares < threshold].index

cleaned_data[prof_diagnosis_type_col] = cleaned_data[prof_diagnosis_type_col].where(
    ~cleaned_data[prof_diagnosis_type_col].isin(rare),
    "other" 
)

cols_to_clean ={
    prof_diagnosis_type_col: 0.01,
    own_diagnosis_type_col: 0.01,
    work_position_col_0: 0.05,  
    work_position_col_1: 0.05,

}

helper.removing_rare_answers(
    cols_to_clean,
    cleaned_data
)

# make all values lowercase
cleaned_data[work_position_col_0] = cleaned_data[work_position_col_0].str.lower()
cleaned_data[work_position_col_1] = cleaned_data[work_position_col_1].str.lower()
    
#%% standardizing gender column

gender_col = "What is your gender?"
print("cleaning gender column")
print("="*40)

print(f"unique values before:")
print(len(cleaned_data[gender_col].unique()))

cleaned_data[gender_col] = (
    cleaned_data[gender_col]
    .astype("string")
    .str.strip()
    .str.lower()
    .map(helper.gender_map(), na_action=None)
    .fillna("lgbtq")
)

print("="*40)
print(f"unique values after:")
print(len(cleaned_data[gender_col].unique()))
print("="*40)
print("values per gender")
print("="*40)
print(cleaned_data[gender_col].value_counts())
print("="*40)

#%% cleaning age column from outliers
print("removing outliers from age column")
age_col = "What is your age?"
cleaned_data[age_col] = np.asarray(
    winsorize(
        cleaned_data[age_col].to_numpy(), 
        limits=0.002
    )
)

print(cleaned_data[age_col].describe(
    percentiles=[0.25,0.5,0.75,0.9]
    )
)

#%% binning the ages, for more
# Define Eurostat-style bins and labels and see if the distribution is ok
print("binning age to reasonable groups")
bins = [0, 24, 34, 44, 54, 64, 100]  
labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

age = pd.DataFrame(cleaned_data[age_col])

# Binning
age["age_group"] = pd.cut(age[age_col], bins=bins, labels=labels, right=True, include_lowest=True)

# Count participants and create percentages per bin
counts = age["age_group"].value_counts().sort_index()
percentages = age["age_group"].value_counts(normalize=True).sort_index() * 100

# print("Counts per bin:\n", counts)
# print("\nPercentages per bin:\n", percentages.round(2))

# we see that the last 2 bins (seasoned people) could be statistically irrelavant (<5%)
# to get a more stable binning, the last bin would be 45+
bins = [0, 24, 34, 44, 100]  
labels = ["<25", "25-34", "35-44", "45+"]

cleaned_data["age_group"] = pd.cut(cleaned_data[age_col], bins=bins, labels=labels, right=True, include_lowest=True)
counts = cleaned_data["age_group"].value_counts().sort_index()
percentages = cleaned_data["age_group"].value_counts(normalize=True).sort_index() * 100

print("Counts per bin:\n", counts)
print("\nPercentages per bin:\n", percentages.round(2))


#%% cleaning country columns
work_col = "What country do you work in?"
live_col = "What country do you live in?"
country = cleaned_data[work_col] == cleaned_data[live_col]

print("="*40)
print("countries where work in = live in")
print(f"{country.mean()*100:2.2f} %")
print("="*40)
print("percentage of countries to the whole dataset")
print("="*40)
print((cleaned_data[work_col].value_counts(normalize=True) * 100).head(10))
# many countries below 5%

# grouping all EU countries together
eu_arr = coco.convert(names=cleaned_data[work_col], to="EU")
cleaned_data[work_col] = np.where(
    np.array(eu_arr) == "EU",
    "European Union",
    cleaned_data[work_col]
)

# grouping everything below 5% to "other"
threshold = 0.05  
shares = cleaned_data[work_col].value_counts(normalize=True)
rare = shares[shares < threshold].index

cleaned_data[work_col] = cleaned_data[work_col].where(
    ~cleaned_data[work_col].isin(rare),
    "other" 
)

print("="*40)
print("final values in dataset")
print("="*40)
print(f"{cleaned_data[work_col].value_counts(normalize=True)*100}")

#%% check if the same is true for territory
terr_work = "What US state or territory do you work in?"
terr_live = "What US state or territory do you live in?"
cleaned_data[terr_work] = cleaned_data[terr_work].fillna("non-us")
cleaned_data[terr_live] = cleaned_data[terr_live].fillna("non-us")

territory = cleaned_data[terr_work] == cleaned_data[terr_live]

print("="*40)
print("territories where work in = live in")
print("="*40)
print(f"{territory.mean()*100:2.2f} %")
print("="*40)
print(cleaned_data[terr_work].value_counts(normalize=True)*100)

#%% dropping uninformative columns
# work in and live territory match with 95% "live in" column can be dropped
cleaned_data = cleaned_data.drop(
    columns=["What US state or territory do you live in?"],
    errors="ignore"
)

# work in and live country match with 98% "live in" column can be dropped
cleaned_data = cleaned_data.drop(
    columns=["What country do you live in?"], 
    errors="ignore"
)

print(f"after cleaning:")
print(cleaned_data.shape)


#%% plotting age, gender, country and territory for US-participants

age_groups = cleaned_data["age_group"]
age_dtype = pd.CategoricalDtype(categories=labels, ordered=True)

gender_sort = ["male", "female", "lgbtq"]
gender_dtype = pd.CategoricalDtype(categories=gender_sort, ordered=True)

age_counts = (
    cleaned_data["age_group"]
    .astype(age_dtype)
    .value_counts()
    .reindex(labels, fill_value=0)
)

gender_counts = (
    cleaned_data[gender_col]
    .astype(gender_dtype)
    .value_counts()
    .reindex(gender_sort, fill_value=0)
)


fig, axes = plt.subplots(1,2, sharey=True, figsize=(8.27,4.5))
axes[0].bar(
    age_counts.index,
    age_counts.values,
)

axes[0].set_ylabel("Count")
axes[0].set_title("Age distribution")

axes[1].bar(
    gender_counts.index,
    gender_counts.values
)
axes[1].set_title("Gender")
axes[1].yaxis.set_visible(False)
plt.tight_layout()
fig.savefig(
    "fig_age_and_gender.jpeg",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
plt.close(fig)

#%%
helper.plot_circle_with_table(
    column=work_col,
    data=cleaned_data,
    title="Country distribution",
    savefile=True,
    savefile_name="fig_country_distribution.jpeg"
)

#%%
#plotting USA territory
# Load a US states geometry (Census 2022 1:20m)
if not os.path.exists("cb_2018_us_state_500k.zip"):
    shp_url = wget.download("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip")
shp_url = "cb_2018_us_state_500k.zip"
states = gpd.read_file(shp_url)[["STUSPS", "NAME", "geometry"]]

us_participant_mask = cleaned_data[terr_work] != "non-us"
us_participants = cleaned_data.loc[us_participant_mask, terr_work].reset_index(drop=True)
print((us_participants.value_counts(normalize=True)*100).head(5))
states = states.rename(columns={"NAME": "territory"})

counts = (
    us_participants
    .value_counts()
    .rename_axis("territory")
    .reset_index(name="count")
)

# Join counts to geodataframe
gdf = states.merge(counts, on="territory", how="left")
gdf["count"] = gdf["count"].fillna(0)

gdf = gdf.to_crs(5070)

# split out smaller territories
lower48 = gdf[~gdf["STUSPS"].isin(["AK","HI","PR","GU","VI","AS","MP"])].copy()
ak = gdf[gdf["STUSPS"]=="AK"].copy()
hi = gdf[gdf["STUSPS"]=="HI"].copy()
pr = gdf[gdf["STUSPS"]=="PR"].copy()

# helper to scale + translate an entire GeoSeries


# bounds of the Lower 48 to position insets relative to it
xmin, ymin, xmax, ymax = lower48.total_bounds
W = xmax - xmin
H = ymax - ymin

# --- Alaska: shrink and park bottom-left ---
if not ak.empty:
    ak.geometry = helper.scale_translate(ak.geometry, xfact=0.35, yfact=0.35)  # shrink
    axmin, aymin, axmax, aymax = ak.total_bounds
    target_xmin = xmin + 0.02*W
    target_ymin = ymin - 0.33*H
    ak_dx = target_xmin - axmin
    ak_dy = target_ymin - aymin
    ak.geometry = helper.scale_translate(ak.geometry, xoff=ak_dx, yoff=ak_dy)

# --- Hawaii: place to the right of Alaska inset ---
if not hi.empty:
    hxmin, hymin, hxmax, hymax = hi.total_bounds
    target_xmin = xmin + 0.22*W
    target_ymin = ymin - 0.28*H
    hi_dx = target_xmin - hxmin
    hi_dy = target_ymin - hymin
    hi.geometry = helper.scale_translate(
        hi.geometry, xfact=1.05, yfact=1.05, xoff=hi_dx, yoff=hi_dy)

# --- Puerto Rico: tuck under Florida area ---
if not pr.empty:
    pxmin, pymin, pxmax, pymax = pr.total_bounds
    target_xmin = xmin + 0.70*W
    target_ymin = ymin - 0.22*H
    pr_dx = target_xmin - pxmin
    pr_dy = target_ymin - pymin
    pr.geometry = helper.scale_translate(
        pr.geometry, 
        xfact=1.20, 
        yfact=1.20, 
        xoff=pr_dx, 
        yoff=pr_dy
    )

# recombine for plotting
gdf_compact = pd.concat([lower48, ak, hi, pr], ignore_index=True)

# territory_fig, ax = plt.subplots()

ax = gdf_compact.plot(
    column="count",
    cmap="Blues",
    linewidth=0.6,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "No. of Participants"},
    missing_kwds={"color": "lightgrey", "hatch": "///", "label": "No data"},
    figsize=(8.27,4.5),
)

ax.set_title("Participants working in the USA", fontsize=14)
ax.axis("off")
territory_fig = ax.get_figure()
territory_fig.savefig(                  #type:ignore
    "fig_participants_by_state.jpg", 
    dpi=300, 
    bbox_inches="tight", 
    facecolor="white"
)
plt.tight_layout()
plt.show()
plt.close(territory_fig)                #type:ignore

#%% check whats left
for col in cleaned_data.columns:
    if cleaned_data[col].dtype != "object":
        print(f"Numerical column: \n {col}")

    if len(cleaned_data[col].value_counts().unique()) > 7:
        print(f"Column with many values: \n {col}")

# all columns are expected to have many unique values


#%% fill all NaN with "not answered"
print("filling NA Values which are not answered")
for col in cleaned_data.columns:
    cleaned_data[col] = (
        cleaned_data[col]
        .astype("string")
        .fillna("not answered")
    )
        
    isNa = cleaned_data[col].isna().sum()
    if isNa > 0:
        print(f"{col} has still NaN values")

#%% plot some demographics
helper.plot_circle_with_table(
    prof_diagnosis_type_col,
    cleaned_data,
    "Professional diagnosis",
    savefile=True,
    savefile_name="fig_professional_diagnosis_0.jpeg"
)

helper.plot_circle_with_table(
    own_diagnosis_type_col,
    cleaned_data,
    "Participants reported self-diagnosis",
    savefile=True,
    savefile_name="fig_own_diagnosis_0.jpeg"
)

helper.plot_circle_with_table(
    work_position_col_0,
    cleaned_data,
    "First Work Position of participants",
    savefile=True,
    savefile_name="fig_work_pos_0.jpeg"
)

helper.plot_circle_with_table(
    work_position_col_1,
    cleaned_data,
    "Second Work Position of participants",
    savefile=True,
    savefile_name="fig_work_pos_1.jpeg"
)









########################
# finished data cleaning
# and demographics description
########################











#%% t-SNE of all data
tsne_all = helper.do_ohe_and_tsne(cleaned_data)

#%% plot
# nice separation, can be shown initially ass it describes the distinct clusters
prof_diagnosed_col = "Have you been diagnosed with a mental health condition by a medical professional?"
searched_help_col = "Have you ever sought treatment for a mental health issue from a mental health professional?"
self_employed_col = "Are you self-employed?"
previous_empl_col = "Do you have previous employers?"
work_col = "What country do you work in?"
currently_having_issue_col = "Do you currently have a mental health disorder?"


#%% self-employed vs. previously employed
helper.plot_tsne_by(self_employed_col, cleaned_data, tsne_all)
helper.plot_tsne_by(previous_empl_col, cleaned_data, tsne_all)

#%% diagnosed vs. searched help
# interesting, as more people sought help than people who are actually diagnosed
helper.plot_tsne_by(prof_diagnosed_col, cleaned_data, tsne_all)
helper.plot_tsne_by(searched_help_col, cleaned_data, tsne_all)

diagnosed = cleaned_data[prof_diagnosed_col] == "Yes"
searched_help = cleaned_data[searched_help_col] == "Yes"

print("="*40)
print(f"people diagnosed with mental health issue: {diagnosed.mean()*100:2.2f} %")
print("="*40)
print(f"people searched professional help: {searched_help.mean()*100:2.2f} %")
print("="*40)

#%% 
# the participants who sought help without being diagnosed answered with "Maybe"
helper.plot_tsne_by(currently_having_issue_col, cleaned_data, tsne_all)

# maybe there is some correlation with remote work?
# maybe there is something regarding workplace and mental health care coverage by employer)
# this should be checked maybe with a linear regressor?
helper.plot_tsne_by("Do you work remotely?", cleaned_data, tsne_all)
helper.plot_tsne_by(work_col, cleaned_data, tsne_all)
helper.plot_tsne_by("Does your employer provide mental health benefits as part of healthcare coverage?", cleaned_data, tsne_all)

# more female and lgbtq diagnosed with mental helath issues?
helper.plot_tsne_by("What is your gender?",cleaned_data, tsne_all)

# people without family history more likely to not have mental health issues?
helper.plot_tsne_by("Do you have a family history of mental illness?", cleaned_data, tsne_all)

# not informative for now:
# plot_tsne_by("age_group",cleaned_data, tsne_all)
# plot_tsne_by("How many employees does your company or organization have?",cleaned_data, tsne_all)

#%% creating 2x2 plot for visualization 
# this shows how the employee-status cluster correspond to the questions
cols_to_show = [
    self_employed_col,
    previous_empl_col,
]

fig, axes = helper.plot_tsne_grid(
    cols_to_show, 
    cleaned_data, 
    tsne_all,
    alt_title = ["Self Employed", "Previous employed"], 
    nrows=1, 
    ncols=2, 
    figsize=(8.27,4.5)
)
fig.savefig(
    "fig_tsne_employed_and_previously_employed.jpeg",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
plt.close(fig)

# this shows that slightly more people sought treatment than the ones who are actually diagnosed
cols_to_show = [
    prof_diagnosed_col,
    searched_help_col,
]

fig, axes = helper.plot_tsne_grid(
    cols_to_show,
    cleaned_data, 
    tsne_all,
    alt_title = [
        "Diagnosed by a professional", 
        "Searched professional help "
    ], 
    nrows=1, 
    ncols=2, 
    figsize=(8.27,4.5)
)
fig.savefig(
    "fig_tsne_diagnosed_vs_got_treatment.jpeg",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
plt.close(fig)

# %% do t-SNE only on employees
employed_only = cleaned_data.copy()

print(f"rows before {employed_only.shape[0]}")
employed_only = employed_only.drop(employed_only[employed_only[self_employed_col] == "Yes"].index)
employed_only = employed_only.drop(labels=self_employed_col, axis=1)

print(f"rows after {employed_only.shape[0]}")

employed_only_tsne = helper.do_ohe_and_tsne(employed_only)

# %% plot 
# here we see maybe an information, why more people sought help without being diagnosed?
helper.plot_tsne_by("Do you have a family history of mental illness?",employed_only, employed_only_tsne)

# maybe there is a tendency, that female and lgbtq people do not suffer as much as male people
# but this is tricky, as we have much more male people
helper.plot_tsne_by("What is your gender?",employed_only, employed_only_tsne)

# EU more likel to have no mental health issues?
helper.plot_tsne_by(work_col,employed_only, employed_only_tsne)

# no more detailed clusters than in the overall plot
# plot_tsne_by(previous_empl_col,employed_only, employed_only_tsne)
# plot_tsne_by(prof_diagnosed_col,employed_only, employed_only_tsne)
# plot_tsne_by(searched_help_col,employed_only, employed_only_tsne)

# not informative
# plot_tsne_by("age_group",employed_only, employed_only_tsne)

# %% do t-SNE only on employees without previous employment and without mental health issue
employed_special_wo = cleaned_data.copy()

print(f"rows before {employed_special_wo.shape[0]}")
# employed
employed_special_wo = employed_special_wo.drop(
    employed_special_wo[employed_special_wo[self_employed_col] == "Yes"].index
)
employed_special_wo = employed_special_wo.drop(labels=self_employed_col, axis=1)

print(f"employed only {employed_special_wo.shape[0]}")
# previous employed
employed_special_wo = employed_special_wo.drop(employed_special_wo[employed_special_wo[previous_empl_col] == "No"].index)
employed_special_wo = employed_special_wo.drop(labels=previous_empl_col, axis=1)

print(f"not prev. employed {employed_special_wo.shape[0]}")

# not diagnosed mental health issue
employed_special_wo = employed_special_wo.drop(employed_special_wo[employed_special_wo[prof_diagnosed_col] == "Yes"].index)
employed_special_wo = employed_special_wo.drop(labels=prof_diagnosed_col, axis=1)
print(f"rows after {employed_special_wo.shape[0]}")
# drop reason columns
reason_col = [
    "Why or why not?",
    "Why or why not?.1"
]
employed_special_wo = employed_special_wo.drop(labels=reason_col, axis=1)
employed_special_tsne = helper.do_ohe_and_tsne(employed_special_wo)
print(employed_special_tsne.shape)


# %% do t-SNE only on employees without previous employment and without mental health issue
employed_special_with = cleaned_data.copy()

print(f"rows before {employed_special_with.shape[0]}")
# employed
employed_special_with = employed_special_with.drop(
    employed_special_with[employed_special_with[self_employed_col] == "Yes"].index
)
employed_special_with = employed_special_with.drop(labels=self_employed_col, axis=1)

print(f"employed only {employed_special_with.shape[0]}")
# previous employed
employed_special_with = employed_special_with.drop(employed_special_with[employed_special_with[previous_empl_col] == "No"].index)
employed_special_with = employed_special_with.drop(labels=previous_empl_col, axis=1)

print(f"not prev. employed {employed_special_with.shape[0]}")

# has diagnosed mental health issue
employed_special_with = employed_special_with.drop(
    employed_special_with[employed_special_with[prof_diagnosed_col] == "No"].index
)
employed_special_with = employed_special_with.drop(labels=prof_diagnosed_col, axis=1)
print(f"rows after {employed_special_with.shape[0]}")
# drop reason columns
reason_col = [
    "Why or why not?",
    "Why or why not?.1"
]
employed_special_with = employed_special_with.drop(labels=reason_col, axis=1)
employed_special_tsne = helper.do_ohe_and_tsne(employed_special_with)
print(employed_special_tsne.shape)

# %% plot --> we don´t see nice clustering in this subset
# helper.plot_tsne_by(searched_help_col,employed_special, employed_special_tsne)
# helper.plot_tsne_by("Do you have a family history of mental illness?",employed_special, employed_special_tsne)
# helper.plot_tsne_by("What is your gender?",employed_special, employed_special_tsne)
# helper.plot_tsne_by("age_group",employed_special, employed_special_tsne)
# helper.plot_tsne_by(work_col,employed_special, employed_special_tsne)

# %% all people from the US
us_people = cleaned_data.copy()
us_people = us_people.drop(us_people[us_people[work_col] == "non-us"].index)
us_people = us_people.drop(labels=work_col, axis=1)

us_people_tsne = helper.do_ohe_and_tsne(us_people)

#%% plot: nothing obvious to see here. All seems the same as the overall data
# plot_tsne_by(employed_col,us_people, us_people_tsne)
# plot_tsne_by(previous_empl_col,us_people, us_people_tsne)
# plot_tsne_by(prof_diagnosed_col, us_people, us_people_tsne)
# plot_tsne_by(searched_help_col",us_people, us_people_tsne)
# plot_tsne_by("Do you have a family history of mental illness?",us_people, us_people_tsne)
# plot_tsne_by(prof_diagnosed_col,us_people, us_people_tsne)
# plot_tsne_by("Does your employer provide mental health benefits as part of healthcare coverage?", us_people, us_people_tsne)








#########################
# t-SNE analysis done
#########################














#%% why do people seek help?
# Did everyone who was diagnosed also search for help?
joint_proba_diag = pd.crosstab(
    cleaned_data[searched_help_col],
    cleaned_data[prof_diagnosed_col],
    margins=True,
    normalize="all"
)

print(joint_proba_diag)

print("="*40)
print(f"diagnosed who searched for help")
print(f"{(joint_proba_diag.loc["Yes","Yes"] / joint_proba_diag.loc["All","Yes"])*100:2.1f} %") # type: ignore
print("="*40)
print(f"searched for help without being diagnosed")
print(f"{(joint_proba_diag.loc["Yes","No"] / joint_proba_diag.loc["All","No"])*100:2.1f} %") # type: ignore
print("="*40)

#%% maybe when they think they have a mental health issue?
joint_proba_think = pd.crosstab(
    cleaned_data[searched_help_col],
    cleaned_data[currently_having_issue_col],
    margins=True,
    normalize="all"
)

print(joint_proba_think)

print("="*40)
print(f"No mental health issue reported or unsure")
print(f"{(joint_proba_think.loc["Yes","Maybe"] + joint_proba_think.loc["Yes","No"])*100:2.1f} %") # type: ignore
print("="*40)

#%% prepare df for linear regression
from sklearn.model_selection import train_test_split

family_history_col = "Do you have a family history of mental illness?"
remote_work_col = "Do you work remotely?"
benefits_col = "Does your employer provide mental health benefits as part of healthcare coverage?"
bring_up_mental_issue_col = "Would you bring up a mental health issue with a potential employer in an interview?"
mhe_in_past_col = "Have you had a mental health disorder in the past?"
age_group_col = "age_group"
negative_seen_col = "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?"
observed_negativity_col ="Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?"
discussed_after_observation_col = "Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?"

columns_for_regeression = [
    searched_help_col,
    prof_diagnosed_col,
    currently_having_issue_col,
    self_employed_col,
    previous_empl_col,
    work_col,
    family_history_col,
    remote_work_col,
    benefits_col,
    bring_up_mental_issue_col,
    mhe_in_past_col,
    age_group_col,
    gender_col,
    observed_negativity_col,
    discussed_after_observation_col
]

reg_df = cleaned_data[columns_for_regeression].copy()

column_to_predict = searched_help_col
X = reg_df.drop(columns=column_to_predict)
y = reg_df[column_to_predict]
y = pd.get_dummies(
    y,
    drop_first=True,
    dtype="int"
)

ohe = OneHotEncoder(
    drop=None,
    sparse_output=False,
    dtype="float64",
    handle_unknown="ignore"
)

categorical_cols = X.select_dtypes(include=["object", "bool", "string"]).columns.tolist()
numerical_columns = X.select_dtypes(include=["float64", "Int64"]).columns.tolist()

transformer = ColumnTransformer(
    transformers=[
        ("cat", ohe, categorical_cols),
        ("num", MinMaxScaler(), numerical_columns)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

ohe_cleaned_data = transformer.fit_transform(reg_df)

print(ohe_cleaned_data.shape)

# Reserve 20% of the data for a hold-out evaluation, keeping class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

    
#%% try ridge classification
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# tuning parameters did not change the classifier in terms of confusion matrix
ridge_classifier = RidgeClassifier(
    alpha=1.0,
    solver="auto",
)

ridge_regressor = Pipeline([
    ("transformer", transformer),
    ("ridge", ridge_classifier)
])

ridge_regressor.fit(X_train, y_train)

# Get predictions on the hold-out set
y_pred = ridge_regressor.predict(X_test)

helper.print_confusion_matrix(
    y_pred,
    y_test
)

# Precision, recall, f1-score, support
print("="*40)
print("Classification Report:")
print("="*40)
print(classification_report(y_test, y_pred, digits=3))

# Access the fitted classifier
ridge = ridge_regressor.named_steps["ridge"]

# Get transformed feature names
feature_names = ridge_regressor.named_steps["transformer"].get_feature_names_out()

# Build a dataframe of coefficients
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": ridge.coef_
})

coef_df["abs_coef"] = coef_df["coefficient"].abs()
top_features = coef_df.sort_values(by="abs_coef", ascending=False).head(10)
feature_importances = helper.get_feature_importance(
    ridge_regressor, 
    X_test, 
    y_test,
)

print("="*40)
print("Feature importances (permutation):")
print("="*40)
print(feature_importances)

print("="*40)
print("Top features)")
print("="*40)
print(top_features)

helper.show_sequential_feature_selection(
    model=ridge_classifier,
    direction="forward",
    transformer=transformer,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

helper.show_sequential_feature_selection(
    model=ridge_classifier,
    direction="backward",
    transformer=transformer,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# we see, that a professional diagnosis relates to seeking help
# as with feature importance, we see, that this is the only important feature


#%% doing Logistic regression
from sklearn.linear_model import LogisticRegression
logit_classifier = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.5,
    random_state=42
)

logit_regressor = Pipeline([
    ("transformer", transformer),
    ("logit", logit_classifier)
])

logit_regressor.fit(X_train, y_train)

# Get predictions on the hold-out set
y_pred = logit_regressor.predict(X_test)

helper.print_confusion_matrix(y_pred, y_test)

# Precision, recall, f1-score, support
print("="*40)
print("Classification Report:")
print("="*40)
print(classification_report(y_test, y_pred, digits=3))

# Access the fitted classifier
logit = logit_regressor.named_steps["logit"]

# Get transformed feature names
feature_names = logit_regressor.named_steps["transformer"].get_feature_names_out()
classes = logit.classes_

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": logit.coef_[0]
})

coef_df["abs_coef"] = coef_df["coefficient"].abs()
top_features = coef_df.sort_values(by="abs_coef", ascending=False).head(10)

feature_importances = helper.get_feature_importance(
    logit_regressor, 
    X_test, 
    y_test
)

print("="*40)
print("Feature importances (permutation):")
print("="*40)
print(feature_importances)

print("="*40)
print("Top features)")
print("="*40)
print(top_features)

helper.show_sequential_feature_selection(
    logit_classifier,
    direction="forward",
    transformer=transformer,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test    
)

helper.show_sequential_feature_selection(
    logit_classifier,
    direction="backward",
    transformer=transformer,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test    
)

# same as ridge regression
    
#%% Linear SVM Classification
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC(
    random_state=42
)

svm_regressor = Pipeline([
    ("transformer", transformer),
    ("svm", svm_classifier)
])

svm_regressor.fit(X_train, y_train)

# Get predictions on the hold-out set
y_pred = svm_regressor.predict(X_test)

helper.print_confusion_matrix(y_pred, y_test)

# Precision, recall, f1-score, support
print("="*40)
print("Classification Report:")
print("="*40)
print(classification_report(y_test, y_pred, digits=3))

# Access the fitted classifier
svm = svm_regressor.named_steps["svm"]

# Get transformed feature names
feature_names = svm_regressor.named_steps["transformer"].get_feature_names_out()
classes = svm.classes_

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": svm.coef_[0]
})

coef_df["abs_coef"] = coef_df["coefficient"].abs()
top_features = coef_df.sort_values(by="abs_coef", ascending=False).head(10)

feature_importances = helper.get_feature_importance(
    svm_regressor, 
    X_test, 
    y_test
)

print("="*40)
print("Feature importances (permutation):")
print("="*40)
print(feature_importances)

print("="*40)
print("Top features)")
print("="*40)
print(top_features)

helper.show_sequential_feature_selection(
    svm_classifier,
    direction="forward",
    transformer=transformer,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

helper.show_sequential_feature_selection(
    svm_classifier,
    direction="backward",
    transformer=transformer,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test    
)

#%% results feature elimination of those models
# all models show what was initially also seen: 
# a professional diagnosis is the most influential feature
# for seeking help. 

# After that the second most important features are 
# past and current mental health issues. 

# the rest seems not very influential













############################
# Regression Classifier done
############################

















# %% import everything for DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# apply it to the employed_special scan, so we skip the overall clusters
# similar pattern as the t-SNE show up if we do not subsample

# subset (A)
set_to_scan_1 = employed_special_wo.copy()
set_to_scan_1 = set_to_scan_1.drop(labels=[age_col, "professional_diagnosis_0"], axis=1)
# diagnosis has to be dropped for isomap (constant)

# subset (B)
set_to_scan_2 = employed_special_with.copy()
set_to_scan_2 = set_to_scan_2.drop(labels=age_col, axis=1)

#%% set up pipelinge

set_config(transform_output="pandas")
set_to_scan = set_to_scan_1

dbscan_ohe = OneHotEncoder(
    drop=None,
    sparse_output=False,
    dtype="float64",
    handle_unknown="ignore"
)

# set_to_scan_1 = 50 , set_to_scan_2 = 70
dbscan_pca = PCA(
    n_components=50,
    svd_solver="auto",
    random_state=42
)

dbscan_transformer = ColumnTransformer(
    transformers=[
        ("cat", dbscan_ohe, set_to_scan.columns)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

dbscan = DBSCAN(
        eps= 0.5,
        min_samples=5
    )

dbscan_pca_pipe = Pipeline([
    ("ohe", dbscan_transformer),
    ("pca", dbscan_pca),
    ("dbscan", dbscan)
])

#%% plot the elbow score to see if there is a value we can choose

pca = Pipeline([("ohe", dbscan_transformer), ("pca", dbscan_pca)]).fit(set_to_scan)
ohe = pca.named_steps["ohe"].fit_transform(set_to_scan)
print(ohe.shape)
pca_space = pca.transform(set_to_scan)
pca_obj = pca.named_steps["pca"]
evr = pca_obj.explained_variance_ratio_

threshold = 0.9
cum = np.cumsum(evr)
print(cum)
n = np.arange(1, len(evr) + 1)

# Small helper: components needed to reach threshold
k = int(np.searchsorted(cum, threshold) + 1)
plt.figure(figsize=(8, 4.5))
plt.step(n, cum, where="mid", label="Cumulative explained variance")
plt.scatter(k, cum[k-2], zorder=3)
plt.axhline(threshold, linestyle="--", linewidth=1, label=f"{int(threshold*100)}% threshold")
plt.axvline(k, linestyle="--", linewidth=1, label=f"k = {k}")
plt.xticks(n)
plt.ylim(0, 1.02)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.show()

# k_guess = 5
# kth = plot_k_distance(pca_space, k_guess)
# not really informative as we do not have a knee for set_to_scan1 and set_to_scan 2

# %% do a generic sweep to see, which valiues can be a good starting point
eps_candidates = np.round(np.linspace(3.0, 10, 11), 2)
ms_candidates = [2, 3, 5, 8, 10, 12, 15, 20]

grid_results = helper.try_dbscan_grid(pca_space, eps_candidates, ms_candidates)
print(grid_results.head(10))

#%% train DBSCAN and plot the result
# set_to_scan_1 eps (4.4) and ms (5) show at 2 of 4 clusters, which could be meaningful
# set_to_scan_2 eps (5.1) and ms (5) show at 2 of 4 clusters, which could be meaningful

best_eps = 4.4
best_ms  = 5
dbscan_pca_pipe.set_params(dbscan__eps=best_eps, dbscan__min_samples=best_ms)
dbscan_pca_pipe.fit(set_to_scan)

labels = dbscan_pca_pipe.named_steps["dbscan"].labels_
n_clusters = len(set(labels))
noise_frac = np.mean(labels == -1)
print(f"DBSCAN clusters: {n_clusters}, noise fraction: {noise_frac:.2%}")

pca2 = PCA(n_components=2, random_state=42).fit_transform(pca_space)
x = pca2.iloc[:, 0].to_numpy() #type: ignore
y = pca2.iloc[:, 1].to_numpy() #type: ignore
c = np.asarray(labels)
plt.figure(figsize=(7, 6))
plt.scatter(x, y, c=c, s=20)
plt.title(f"DBSCAN clusters in PCA-2D (eps={best_eps}, min_samples={best_ms})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()


# %% check cluster size and profile back to the ohe data
unique, counts = np.unique(labels, return_counts=True)

clustered = pd.DataFrame(dbscan_ohe.fit_transform(set_to_scan))
clustered["cluster"] = labels
global_means_full = clustered.drop(columns=["cluster"]).mean()
cluster_profiles = clustered.groupby("cluster").mean().T
# cluster_profiles.head(20)
# print(global_means_full)
unique_values_dbscan = clustered["cluster"].value_counts()
print(unique_values_dbscan)

#%% df_with_labels is OHE dataframe + "cluster" column
results = helper.top_features_per_cluster(clustered, cluster_col="cluster", top_n=10)

for cluster in range(len(results)-1):
    print(f"features in {cluster}")
    print(results[cluster])

# %% plot heat map to see, which answers differ in the clusters
candidate_feats = list(cluster_profiles.index)            
clusters = sorted(cluster_profiles.columns)               

helper.plot_heatmap(
    result_df=cluster_profiles, 
    feature_names=candidate_feats, 
    cluster_ids=clusters,
    global_means=global_means_full, 
    top_n=15,
    q_width=25,
    q_splits=3
)

# investigate further cluster 0 and 1
# there seem to be 2 groups with meaning

#%% investigate cluster 0 and 1 (for set_to_scan_1 and 2)

interesting_cluster = cluster_profiles.copy().drop([2,3], axis=1)     # additionally cluster 3 in set_to_scan_1 
interesting_cluster["cluster_0_diff_noise"] = abs(interesting_cluster[0] - interesting_cluster[-1])
interesting_cluster["cluster_1_diff_noise"] = abs(interesting_cluster[1] - interesting_cluster[-1])
interesting_cluster["cluster_1_diff_0"] = abs(interesting_cluster[1] - interesting_cluster[0])
interesting_cluster = interesting_cluster.sort_values(by="cluster_1_diff_0", ascending=False)

print(interesting_cluster.head(10))

# main differences between the 2 main clusters
filter_threshold = 0.5
columns_to_filter = ["cluster_1_diff_0", "cluster_1_diff_noise", "cluster_0_diff_noise"]

for column in columns_to_filter:
    filtered_clusters = interesting_cluster[interesting_cluster[column] >= filter_threshold]
    print("="*40)
    print(filtered_clusters[[column]])
    print("-"*40)
    for index in filtered_clusters.index:
        print(index)

# A)
# cluster 1 are people who are really afraid of sharing their mental health issue
# they won´t discuss this with anyone in the organization

# %%
short_labels = {
    "Do you feel that being identified as a person with a mental health issue would hurt your career?_Yes, I think it would": "Hurting career if identified",
    "Would you bring up a mental health issue with a potential employer in an interview?_No": "No disclosure in interview",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?_No": "Uncomfortable discussing with coworkers",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?_No": "Uncomfortable discussing with supervisor",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?_Yes, I think they would": "Peers view negatively",
}

selected_features = list(short_labels.keys())
renamed_features = [short_labels[f] for f in selected_features]
plot_data = cluster_profiles.loc[selected_features, [-1, 0, 1]]
plot_data.index = pd.Index(renamed_features)

df_radar = plot_data.T
df_radar.index = pd.Index(["Noise", "Cluster 0", "Cluster 1"])
cluster_counts = {"Cluster 0": unique_values_dbscan[0], "Cluster 1": unique_values_dbscan[1], "Noise": unique_values_dbscan[-1]}

helper.plot_radar_with_noise(
    df_radar, 
    cluster_counts, 
    "(A-PCA) Cluster differences in perceived stigma and disclosure comfort",
    True, 
    "fig_radar_dbscan_pca_A.jpeg"
)

#%% shortened DBSCAN for(B)
set_config(transform_output="pandas")
set_to_scan = set_to_scan_2

dbscan_pca = PCA(
    n_components=70,
    svd_solver="auto",
    random_state=42
)

dbscan_pca_pipe = Pipeline([
    ("ohe", dbscan_transformer),
    ("pca", dbscan_pca),
    ("dbscan", dbscan)
])

pca = Pipeline([("ohe", dbscan_transformer), ("pca", dbscan_pca)]).fit(set_to_scan)
pca_space = pca.transform(set_to_scan)
grid_results = helper.try_dbscan_grid(pca_space, eps_candidates, ms_candidates)
print(grid_results.head(10))


#%%
dbscan_pca_pipe.set_params(dbscan__eps=5.1, dbscan__min_samples=8)
dbscan_pca_pipe.fit(set_to_scan)

labels = dbscan_pca_pipe.named_steps["dbscan"].labels_
clustered = pd.DataFrame(dbscan_ohe.fit_transform(set_to_scan))
clustered["cluster"] = labels
cluster_profiles = clustered.groupby("cluster").mean().T
unique_values_dbscan = clustered["cluster"].value_counts()
print(unique_values_dbscan)

# Define which clusters to compare
analysis = helper.analyze_cluster_differences(cluster_profiles, dbscan_ohe)
print(analysis.head(10).index)
short_labels = {
    'Would you feel comfortable discussing a mental health disorder with your coworkers?_No':"Uncomfortable discussing with coworkers",
    'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?_Yes, I think they would':"Peers view negatively",
    'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?_No':"Uncomfortable discussing with supervisor",
    'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?_No':"No active communication from employer",
    'Do you think that discussing a mental health disorder with your employer would have negative consequences?_No':"Disclosure would have no neg. consequences",   
    'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?_None of them':"Not heard about neg. consequences previously",
}


# cluster 2 and 3 excluded as they only have 11 and 6 participants and are not informative
selected_features = list(short_labels.keys())
renamed_features = [short_labels[f] for f in selected_features]
plot_data = cluster_profiles.loc[selected_features, [
    -1, 
    0, 
    1, 
]]
plot_data.index = pd.Index(renamed_features)

df_radar = plot_data.T
df_radar.index = pd.Index([
    "Noise", 
    "Cluster 0", 
    "Cluster 1",
])
cluster_counts = {
    "Cluster 0": unique_values_dbscan[0], 
    "Cluster 1": unique_values_dbscan[1], 
    "Noise": unique_values_dbscan[-1]
}

helper.plot_radar_with_noise(
    df_radar, 
    cluster_counts, 
    "(B-PCA) Cluster differences in perceived stigma and disclosure comfort",
    True, 
    "fig_radar_dbscan_pca_B.jpeg"
)

#%% do the same with Isomap, start with (B)
set_config(transform_output="default")
set_to_scan = set_to_scan_2

# set_to_scan_2 comp=3 , neigh=5
dbscan_iso = Isomap(
    n_components=3,
    n_neighbors=5,
)

dbscan_iso_pipe = Pipeline([
    ("ohe", dbscan_transformer),
    ("iso", dbscan_iso),
    ("dbscan", dbscan)
])

# pandas df causes issues with isomap reconstruction error


iso = Pipeline([("ohe", dbscan_transformer), ("iso", dbscan_iso)]).fit(set_to_scan)
ohe = iso.named_steps["ohe"].fit_transform(set_to_scan)
iso_space = dbscan_iso_pipe.named_steps["iso"].transform(
    dbscan_transformer.transform(set_to_scan)
)

#%% scan for usefull Isomap settings
neighbors_grid = [5, 10, 15, 20, 30, 40, 50, 100]
components_grid = [2, 3, 5]

# Collect results
results = []

for n_components in components_grid:
    for n_neighbors in neighbors_grid:
        try:
            iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
            iso.fit(ohe)
            err = iso.reconstruction_error()
            results.append((n_components, n_neighbors, err))
        except Exception as e:
            print(f"Failed for n_components={n_components}, n_neighbors={n_neighbors}: {e}")
            results.append((n_components, n_neighbors, np.nan))

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=["n_components", "n_neighbors", "reconstruction_error"])
best = results_df.sort_values("reconstruction_error").head(10)
print("Top candidates:")
print(best)
for nc in components_grid:
    subset = results_df[results_df["n_components"] == nc]
    plt.plot(subset["n_neighbors"], subset["reconstruction_error"], marker="o", label=f"{nc} components")

plt.xlabel("n_neighbors")
plt.ylabel("Reconstruction Error")
plt.title("Isomap Diagnostics")
plt.legend()
plt.grid(True)
plt.show()

# elbow suggests using something like 30 neighbors to choose for Isomap
# check which performs better with DBSCAN

#%% 
eps_candidates = np.round(np.linspace(3.0, 10, 11), 2)
ms_candidates = [2, 3, 5, 8, 10, 12, 15, 20]

grid_results = helper.try_dbscan_grid(iso_space, eps_candidates, ms_candidates)
print(grid_results.head(10))

#%%
# set_to_scan_1 eps= , ms= ()
# set_to_scan_2 eps=3 , ms=15
best_eps = 3.7
best_ms  = 20
dbscan_iso_pipe.set_params(dbscan__eps=best_eps, dbscan__min_samples=best_ms)
dbscan_iso_pipe.fit(set_to_scan)

labels = dbscan_iso_pipe.named_steps["dbscan"].labels_
n_clusters = len(set(labels))
noise_frac = np.mean(labels == -1)
print(f"DBSCAN clusters: {n_clusters}, noise fraction: {noise_frac:.2%}")

labels = dbscan_iso_pipe.named_steps["dbscan"].labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_frac = np.mean(labels == -1)

print(f"DBSCAN clusters: {n_clusters}, noise fraction: {noise_frac:.2%}")

iso_df = pd.DataFrame(iso_space, columns=[f"iso_{i}" for i in range(iso_space.shape[1])])
iso_df["label"] = labels

if iso_space.shape[1] > 2:
    sns.pairplot(
        iso_df,
        vars=[f"iso_{i}" for i in range(min(5, iso_space.shape[1]))],
        hue="label",
        palette="tab10",
        corner=True,
        plot_kws={"s": 15, "alpha": 0.7}
    )
    plt.suptitle("Pairwise Plots of Isomap Dimensions with DBSCAN Labels", y=1.02)
    plt.tight_layout()
    plt.show()

unique, counts = np.unique(labels, return_counts=True)

clustered = pd.DataFrame(dbscan_ohe.fit_transform(set_to_scan))
clustered["cluster"] = labels

global_means_full = clustered.drop(columns=["cluster"]).mean()
cluster_profiles = clustered.groupby("cluster").mean().T
# print(cluster_profiles.head(20))
# print(global_means_full)
unique_values_dbscan = clustered["cluster"].value_counts()
print(unique_values_dbscan)

#%%
clustered_df = set_to_scan.copy()
clustered_df["cluster"] = labels

for col in clustered_df.columns.drop("cluster"):
    ct = pd.crosstab(clustered_df["cluster"], clustered_df[col], normalize="index")
    # print(f"\n=== {col} ===")
    # display(ct.round(2))


#%% df_with_labels is OHE dataframe + "cluster" column
results = helper.top_features_per_cluster(clustered, cluster_col="cluster", top_n=10)

for cluster in range(len(results)-1):
    print(f"features in {cluster}")
    print(results[cluster])

candidate_feats = list(cluster_profiles.index)            
clusters = sorted(cluster_profiles.columns)               

#%%

analyzed = helper.analyze_cluster_differences(cluster_profiles, dbscan_ohe)
print(analyzed.head(10).index)


# %% Plotting (B) 

short_labels = {
    'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?_Yes':"Comfortable discussing with supervisor",
    'Do you think that discussing a mental health disorder with your employer would have negative consequences?_No':"Disclosure would have no neg. consequences",
    'Do you feel that your employer takes mental health as seriously as physical health?_Yes':"Employer takes all health issues seriously",
    'Does your employer offer resources to learn more about mental health concerns and options for seeking help?_Yes':"Employer offer options for help",
    'Would you feel comfortable discussing a mental health disorder with your coworkers?_No':"Uncomfortable discussing with coworkers",
    'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?_No':"Uncomfortable discussing with supervisor",
    'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?_Yes, I think they would':"Peers view negatively",
    'Do you feel that being identified as a person with a mental health issue would hurt your career?_Yes, I think it would':"Hurting career if identified",
    'Do you think that discussing a mental health disorder with your employer would have negative consequences?_Yes':"(Current employer) Disclosure would have neg. consequences",
    'Do you think that discussing a mental health disorder with previous employers would have negative consequences?_Yes, all of them':"(Prev. employer) Disclosure would have neg. consequences",
}

# cluster 2 does not differ much from noise --> excluded
# for clarity split into 2 plots
selected_features = list(short_labels.keys())
renamed_features = [short_labels[f] for f in selected_features]
plot_data = cluster_profiles.loc[selected_features, [
    -1, 
    0,
    1,
]]
plot_data.index = pd.Index(renamed_features)

df_radar = plot_data.T
df_radar.index = pd.Index([
    "Noise", 
    "Cluster 0", 
    "Cluster 1", 
])
cluster_counts = {
    "Noise": unique_values_dbscan[-1],
    "Cluster 0": unique_values_dbscan[0], 
    "Cluster 1": unique_values_dbscan[1], 
}

# radar plot looks too crowded
helper.plot_cluster_barplot(
    df_radar=df_radar,
    cluster_counts=cluster_counts,
    title="(B-ISO) Cluster differences in disclosure comfort and employer attitudes",
    savefile=True,
    savefile_name="fig_hbar_dbscan_isomap_B.jpeg"
)

# %% Plotting of Cluster 0 and 3
# comparable with PCA DBSCAN, but less participants in cluster 0
short_labels = {
    "Do you feel that being identified as a person with a mental health issue would hurt your career?_Yes, I think it would": "Hurting career if identified",
    "Would you bring up a mental health issue with a potential employer in an interview?_No": "No disclosure in interview",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?_No": "Uncomfortable discussing with coworkers",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?_No": "Uncomfortable discussing with supervisor",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?_Yes, I think they would": "Peers view negatively",  
}

selected_features = list(short_labels.keys())
renamed_features = [short_labels[f] for f in selected_features]
plot_data = cluster_profiles.loc[selected_features, [
    -1, 
    1
]]
plot_data.index = pd.Index(renamed_features)

df_radar = plot_data.T
df_radar.index = pd.Index([
    "Noise", 
    "Cluster 1", 
])
cluster_counts = {
    "Noise": unique_values_dbscan[-1],
    "Cluster 1": unique_values_dbscan[1], 
}

helper.plot_radar_with_noise(
    df_radar, 
    cluster_counts, 
    "(B-ISO) Cluster differences in perceived stigma and disclosure comfort",
    True, 
    "fig_radar_dbscan_isomap_B.jpeg"
)

#%% shorten DBSCAN for (A)
set_config(transform_output="default")

set_to_scan = set_to_scan_1

iso = Pipeline([("ohe", dbscan_transformer), ("iso", Isomap(n_components=3,n_neighbors=5))]).fit(set_to_scan)
ohe = iso.named_steps["ohe"].fit_transform(set_to_scan)
iso_space = dbscan_iso_pipe.named_steps["iso"].transform(
    dbscan_transformer.transform(set_to_scan)
)

eps_candidates = np.round(np.linspace(3.0, 10, 11), 2)
ms_candidates = [2, 3, 5, 8, 10, 12, 15, 20]

grid_results = helper.try_dbscan_grid(iso_space, eps_candidates, ms_candidates)
print(grid_results.head(10))

#%% 
dbscan_iso_pipe.set_params(dbscan__eps=3, dbscan__min_samples=12)
dbscan_iso_pipe.fit(set_to_scan)
labels = dbscan_iso_pipe.named_steps["dbscan"].labels_

clustered = pd.DataFrame(dbscan_ohe.fit_transform(set_to_scan))
clustered["cluster"] = labels
cluster_profiles = clustered.groupby("cluster").mean().T
ohe_feature_names = dbscan_transformer.get_feature_names_out()
cluster_profiles.index = pd.Index(ohe_feature_names)
unique_values_dbscan = clustered["cluster"].value_counts()
print(unique_values_dbscan)

analyzed = helper.analyze_cluster_differences(cluster_profiles, dbscan_ohe, True)

#%% plotting for set_to_scan_1 (A)
short_labels = {    
    "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?_No, at none of my previous employers": "Uncomfortable discussing with prev. supervisor",  
    "Would you feel comfortable discussing a mental health disorder with your coworkers?_No":"Uncomfortable discussing with coworkers",
    "Did your previous employers provide resources to learn more about mental health issues and how to seek help?_None did":"No ressources provided formerly",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?_Yes, I think they would":"Peers view negatively",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?_No":"Uncomfortable discussing with supervisor",
    "Have your previous employers provided mental health benefits?_No, none did": "No previous benefits",
}

selected_features = list(short_labels.keys())
renamed_features = [short_labels[f] for f in selected_features]
plot_data = cluster_profiles.loc[selected_features, [
    -1, 
    0, 
    1,
]]
plot_data.index = pd.Index(renamed_features)

df_radar = plot_data.T
df_radar.index = pd.Index([
    "Noise", 
    "Cluster 0", 
    "Cluster 1", 
])
cluster_counts = {
    "Noise": unique_values_dbscan[-1],
    "Cluster 0": unique_values_dbscan[0], 
    "Cluster 1": unique_values_dbscan[1], 
}

helper.plot_radar_with_noise(
    df_radar, 
    cluster_counts, 
    "(A-ISO) Cluster differences in perceived stigma and disclosure comfort",
    True, 
    "fig_radar_dbscan_isomap_A.jpeg"
)


#%% check correlation between knowing health benefits and supervisor disclosure
from scipy.stats import chi2_contingency

clustered = pd.DataFrame(dbscan_ohe.fit_transform(cleaned_data))

contingency = pd.crosstab(
    clustered["Do you know the options for mental health care available under your employer-provided coverage?_No"],
    clustered["Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?_Yes"]
)

print(contingency)
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi² = {chi2:.3f}, p = {p:.4f}")
 
# Yes/Yes: Chi² = 5.837, p = 0.0157
# No/No: Chi² = 6.568, p = 0.0104
# Yes/No: Chi² = 2.209, p = 0.1372
# No/Yes: Chi² = 6.426, p = 0.0112

# Who is informed also talks to supervisor and vice versa
# small confident group (31) have trust and talk to supervisor without knowing about benefits.

#%%






##############################
# DBSCAN done
##############################















# %% are certain working positions more prevalent for mental health issue?
melted = cleaned_data.melt(
    id_vars=prof_diagnosed_col,
    value_vars=[work_position_col_0, work_position_col_1],
    value_name="merged_cat"
)[[prof_diagnosed_col, "merged_cat"]]

crosstab_prof = pd.crosstab(
    melted["merged_cat"], 
    melted[prof_diagnosed_col],
    normalize="index"
)*100
print(crosstab_prof.round(2))

# seems that support is more vulnerable to MHI 
# other and designer have a tendency to be mor vulnerable

melted = cleaned_data.melt(
    id_vars=currently_having_issue_col,
    value_vars=[work_position_col_0, work_position_col_1],
    value_name="merged_cat"
)[[currently_having_issue_col, "merged_cat"]]


crosstab_self = pd.crosstab(
    melted["merged_cat"], 
    melted[currently_having_issue_col],
    normalize="index"
)*100

crosstab_self["Yes/Maybe"] = crosstab_self["Maybe"] + crosstab_self["Yes"]
crosstab_self = crosstab_self.drop(["Maybe", "Yes"], axis=1)

print(crosstab_self.round(2))

# self reported MHI seems to show that all groups have only 1/3 answered with "No"

# %% Why would participants not bring up MHI in interviews?
physical_interview_col = "Would you be willing to bring up a physical health issue with a potential employer in an interview?"
reason_physical_interview_col = "Why or why not?"
mental_interview_col = "Would you bring up a mental health issue with a potential employer in an interview?"
reason_mental_interview_col = "Why or why not?.1"

physical_mental_crosstab = pd.crosstab(
    cleaned_data[physical_interview_col],
    cleaned_data[mental_interview_col],
    normalize=True,
    margins=True
)*100

print(physical_mental_crosstab)
print(cleaned_data[mental_interview_col].value_counts())
# %%
# masking physical interview
physical_mask_dict = {
    "Yes":"Would bring up physical issus in interview",
    "No":"Would not bring up physical health issue in interview",
    "Maybe":"Would maybe bring up physical health issue in interview",
}

physical_df_list = helper.create_word_counts(
    reason_physical_interview_col, 
    physical_interview_col, 
    physical_mask_dict,
    cleaned_data,
    export=True
)

for df, headings in zip(physical_df_list, physical_mask_dict.values()):
    print("="*50)
    print(headings)
    print("="*50)
    print(df)

    helper.create_table_in_word(df, headings)

print(cleaned_data[physical_interview_col].value_counts())

# %%
menatl_mask_dict = {
    "Yes":"Would bring up mental issus in interview",
    "No":"Would not bring up mental health issue in interview",
    "Maybe":"Would maybe bring up mental health issue in interview",
}
mental_df_list = helper.create_word_counts(
    reason_mental_interview_col, 
    mental_interview_col, 
    menatl_mask_dict,
    cleaned_data,
    export=True
)

for df, headings in zip(mental_df_list, menatl_mask_dict.values()):
    print("="*50)
    print(headings)
    print("="*50)
    print(df)

    helper.create_table_in_word(df, headings)


# %%
