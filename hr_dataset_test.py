"""
HR Candidate Ranking using Group VIKOR

This project demonstrates candidate evaluation and ranking
using the Group VIKOR multi-criteria decision-making method.

The analysis is performed on a real-world HR analytics dataset,
where multiple decision makers define evaluation criteria and weights.

The project includes:
- Data preprocessing
- Feature transformation
- Group weighting
- VIKOR ranking
- Data visualization
- Excel report generation

Author: Maria Rodopoulou
"""
import pandas as pd

df = pd.read_csv(r"C:\Users\maria\Desktop\Agent.Allocator.Report\aug_train.csv")

print(df.head())
print(df.columns)
# Convert categorical variables into numerical scores

education_map = {
    "Primary School": 1,
    "High School": 2,
    "Graduate": 3,
    "Masters": 4,
    "Phd": 5
}

company_size_map = {
    "<10": 1,
    "10/49": 2,
    "50-99": 3,
    "100-500": 4,
    "500-999": 5,
    "1000-4999": 6,
    "5000-9999": 7,
    "10000+": 8
}

experience_map = {
    "<1": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "11": 11,
    "12": 12,
    "13": 13,
    "14": 14,
    "15": 15,
    "16": 16,
    "17": 17,
    "18": 18,
    "19": 19,
    "20": 20,
    ">20": 21
}

df["education_score"] = df["education_level"].map(education_map)
df["company_size_score"] = df["company_size"].map(company_size_map)
df["experience_score"] = df["experience"].map(experience_map)

criteria = [
    "training_hours",
    "city_development_index",
    "experience_score",
    "education_score",
    "company_size_score"
]

decision_makers = {
    "HR Manager": [0.25, 0.15, 0.25, 0.25, 0.10],
    "Technical Lead": [0.20, 0.10, 0.35, 0.25, 0.10],
    "CEO": [0.15, 0.25, 0.20, 0.20, 0.20]
}

benefit_cost = {
    "training_hours": "benefit",
    "city_development_index": "benefit",
    "experience_score": "benefit",
    "education_score": "benefit",
    "company_size_score": "benefit"
}

print(df[criteria].head())
print(decision_makers)
import numpy as np

# =========================
# GROUP VIKOR
# =========================
# Replace missing values with median values
for c in criteria:
    df[c] = df[c].fillna(df[c].median())

# Create decision matrix
matrix = df[criteria].astype(float).values

# Collect weights from all decision makers
all_weights = np.array(list(decision_makers.values()))

# Compute average group weights
group_weights = np.mean(all_weights, axis=0)

print("\nGroup Weights:")
print(group_weights)

# =========================
# Compute best and worst values
# =========================

f_star = []
f_minus = []

for i, criterion in enumerate(criteria):

    if benefit_cost[criterion] == "benefit":
        f_star.append(matrix[:, i].max())
        f_minus.append(matrix[:, i].min())

    else:
        f_star.append(matrix[:, i].min())
        f_minus.append(matrix[:, i].max())

f_star = np.array(f_star)
f_minus = np.array(f_minus)

# =========================
# S and R
# =========================

S = []
R = []

for row in matrix:

    values = []

    for j in range(len(criteria)):

        value = group_weights[j] * (
            (f_star[j] - row[j]) /
            (f_star[j] - f_minus[j] + 1e-9)
        )

        values.append(value)

    S.append(sum(values))
    R.append(max(values))

S = np.array(S)
R = np.array(R)

# =========================
# Q values
# =========================

v = 0.5

S_star = S.min()
S_minus = S.max()

R_star = R.min()
R_minus = R.max()

Q = []

for i in range(len(matrix)):

    q = v * ((S[i] - S_star) / (S_minus - S_star + 1e-9)) + \
        (1 - v) * ((R[i] - R_star) / (R_minus - R_star + 1e-9))

    Q.append(q)

Q = np.array(Q)

# =========================
# Results
# =========================

results = df.copy()

results["Q"] = Q
results["S"] = S
results["R"] = R

results = results.sort_values(by="Q")

results["Rank"] = range(1, len(results) + 1)

print("\nTOP 10 Candidates:")
print(results[
    [
        "enrollee_id",
        "training_hours",
        "experience_score",
        "education_score",
        "S",
        "R",
        "Q",
        "Rank"
    ]
].head(10))

# =========================
# Export Excel
# =========================

import matplotlib.pyplot as plt

output_file = r"C:\Users\maria\Desktop\Agent.Allocator.Report\HR_Candidate_Ranking_GroupVIKOR.xlsx"

weights_df = pd.DataFrame({
    "Criterion": criteria,
    "Group Weight": group_weights
})

top_candidates = results.head(20)

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Raw_HR_Data", index=False)
    weights_df.to_excel(writer, sheet_name="CriteriaWeights", index=False)
    results.to_excel(writer, sheet_name="AllCandidates_VIKOR", index=False)
    top_candidates.to_excel(writer, sheet_name="TopCandidates", index=False)

    workbook = writer.book

    # Chart 1: Criteria weights
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(weights_df["Criterion"], weights_df["Group Weight"])
    ax1.set_title("Group Weights per Criterion")
    ax1.set_ylabel("Weight")
    ax1.tick_params(axis="x", rotation=45)
    fig1.tight_layout()

    chart1_path = "HR_Weights_Chart.png"
    fig1.savefig(chart1_path)

    worksheet1 = workbook.add_worksheet("Weights_Chart")
    writer.sheets["Weights_Chart"] = worksheet1
    worksheet1.insert_image("B2", chart1_path)

    # Chart 2: Top candidates Q scores
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(top_candidates["enrollee_id"].astype(str), top_candidates["Q"])
    ax2.set_title("Top Candidates - VIKOR Q Scores")
    ax2.set_ylabel("Q Score")
    ax2.set_xlabel("Candidate ID")
    ax2.tick_params(axis="x", rotation=45)
    fig2.tight_layout()

    chart2_path = "HR_Top_Candidates_Q_Chart.png"
    fig2.savefig(chart2_path)

    worksheet2 = workbook.add_worksheet("Q_Scores_Chart")
    writer.sheets["Q_Scores_Chart"] = worksheet2
    worksheet2.insert_image("B2", chart2_path)

print(f"\n✅ Output file generated {output_file}")