import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Load & Clean ───────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/BankChurners.csv')
df.columns = df.columns.str.strip()

# Drop Naive Bayes classifier columns
df = df[[c for c in df.columns if 'Naive_Bayes' not in c]]

# Binary churn flag
df['Churned'] = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)

# ══════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════

total = len(df)
churned = df['Churned'].sum()
churn_rate = round(churned / total * 100, 2)

# 1. Churn by Income Category
inc_churn = df.groupby('Income_Category').agg(
    total=('Churned','count'),
    churned=('Churned','sum')
).reset_index()
inc_churn['churn_rate'] = (inc_churn['churned'] / inc_churn['total'] * 100).round(2)
inc_order = ['Less than $40K','$40K - $60K','$60K - $80K','$80K - $120K','$120K +','Unknown']
inc_churn = inc_churn.set_index('Income_Category').reindex(inc_order).dropna().reset_index()

# 2. Churn by Card Category
card_churn = df.groupby('Card_Category').agg(
    total=('Churned','count'),
    churned=('Churned','sum')
).reset_index()
card_churn['churn_rate'] = (card_churn['churned'] / card_churn['total'] * 100).round(2)
card_churn = card_churn.sort_values('churn_rate', ascending=False)

# 3. Inactive Months vs Churn
inact_churn = df.groupby('Months_Inactive_12_mon').agg(
    total=('Churned','count'),
    churned=('Churned','sum')
).reset_index()
inact_churn['churn_rate'] = (inact_churn['churned'] / inact_churn['total'] * 100).round(2)

# 4. Transaction Count vs Churn (binned)
df['trans_band'] = pd.cut(df['Total_Trans_Ct'],
    bins=[0,20,40,60,80,150],
    labels=['1-20','21-40','41-60','61-80','81+'])
trans_churn = df.groupby('trans_band', observed=True).agg(
    total=('Churned','count'),
    churned=('Churned','sum')
).reset_index()
trans_churn['churn_rate'] = (trans_churn['churned'] / trans_churn['total'] * 100).round(2)

# 5. Relationship Count vs Churn
rel_churn = df.groupby('Total_Relationship_Count').agg(
    total=('Churned','count'),
    churned=('Churned','sum')
).reset_index()
rel_churn['churn_rate'] = (rel_churn['churned'] / rel_churn['total'] * 100).round(2)

# 6. High Risk Segment
high_risk = df[
    (df['Months_Inactive_12_mon'] >= 3) &
    (df['Total_Trans_Ct'] < 40) &
    (df['Total_Relationship_Count'] <= 2)
]
hr_total = len(high_risk)
hr_churned = high_risk['Churned'].sum()
hr_rate = round(hr_churned / hr_total * 100, 2) if hr_total > 0 else 0

# 7. Avg transaction amount: churned vs retained
avg_trans = df.groupby('Attrition_Flag')['Total_Trans_Amt'].mean().round(2)

# ══════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
NAVY  = '#1B2A4A'
BLUE  = '#2E75B6'
LIGHT = '#A8C4E0'
RED   = '#C0392B'
GREEN = '#1A7A4A'
GRAY  = '#F7F9FC'

fig = plt.figure(figsize=(18, 22))
fig.patch.set_facecolor(GRAY)

fig.text(0.5, 0.975, 'Bank Customer Churn — Consulting Analytics Report',
         ha='center', fontsize=18, fontweight='bold', color=NAVY)
fig.text(0.5, 0.960, 'Credit Card Customers Dataset  |  10,127 Customers  |  Python EDA + Business Intelligence',
         ha='center', fontsize=11, color='#555555')

def make_ax(left, bottom, width, height):
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_facecolor('white')
    for s in ax.spines.values():
        s.set_edgecolor('#DDDDDD')
    return ax

ax1 = make_ax(0.05, 0.735, 0.40, 0.195)
ax2 = make_ax(0.55, 0.735, 0.40, 0.195)
ax3 = make_ax(0.05, 0.490, 0.40, 0.195)
ax4 = make_ax(0.55, 0.490, 0.40, 0.195)
ax5 = make_ax(0.05, 0.245, 0.40, 0.195)
ax6 = make_ax(0.55, 0.245, 0.40, 0.195)

# Plot 1: Churn by Income
colors1 = [RED if x == inc_churn['churn_rate'].max() else BLUE for x in inc_churn['churn_rate']]
bars1 = ax1.barh(inc_churn['Income_Category'], inc_churn['churn_rate'], color=colors1, height=0.55)
ax1.set_title('Churn Rate by Income Category', fontweight='bold', color=NAVY, fontsize=11)
ax1.set_xlabel('Churn Rate (%)', fontsize=9)
for bar, val in zip(bars1, inc_churn['churn_rate']):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val}%', va='center', fontsize=9, fontweight='bold', color=NAVY)
ax1.set_xlim(0, inc_churn['churn_rate'].max() + 8)
ax1.tick_params(axis='y', labelsize=8)

# Plot 2: Churn by Card Category
colors2 = [RED if x == card_churn['churn_rate'].max() else BLUE for x in card_churn['churn_rate']]
bars2 = ax2.bar(card_churn['Card_Category'], card_churn['churn_rate'], color=colors2, width=0.45)
ax2.set_title('Churn Rate by Card Category', fontweight='bold', color=NAVY, fontsize=11)
ax2.set_ylabel('Churn Rate (%)', fontsize=9)
for bar, val in zip(bars2, card_churn['churn_rate']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val}%', ha='center', fontsize=9, fontweight='bold', color=NAVY)
ax2.set_ylim(0, card_churn['churn_rate'].max() + 10)

# Plot 3: Inactive Months vs Churn
colors3 = [RED if x >= 3 else BLUE for x in inact_churn['Months_Inactive_12_mon']]
bars3 = ax3.bar(inact_churn['Months_Inactive_12_mon'].astype(str),
                inact_churn['churn_rate'], color=colors3, width=0.55)
ax3.set_title('Churn Rate by Months Inactive (Last 12M)', fontweight='bold', color=NAVY, fontsize=11)
ax3.set_xlabel('Months Inactive', fontsize=9)
ax3.set_ylabel('Churn Rate (%)', fontsize=9)
for bar, val in zip(bars3, inact_churn['churn_rate']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val}%', ha='center', fontsize=8, fontweight='bold', color=NAVY)
ax3.set_ylim(0, inact_churn['churn_rate'].max() + 12)
red_patch = mpatches.Patch(color=RED, label='High Risk (3+ months)')
blue_patch = mpatches.Patch(color=BLUE, label='Lower Risk')
ax3.legend(handles=[red_patch, blue_patch], fontsize=8)

# Plot 4: Transaction Count vs Churn
colors4 = [RED if x == trans_churn['churn_rate'].max() else BLUE for x in trans_churn['churn_rate']]
bars4 = ax4.bar(trans_churn['trans_band'].astype(str), trans_churn['churn_rate'],
                color=colors4, width=0.5)
ax4.set_title('Churn Rate by Transaction Count Band', fontweight='bold', color=NAVY, fontsize=11)
ax4.set_xlabel('Transaction Count Range', fontsize=9)
ax4.set_ylabel('Churn Rate (%)', fontsize=9)
for bar, val in zip(bars4, trans_churn['churn_rate']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}%', ha='center', fontsize=9, fontweight='bold', color=NAVY)
ax4.set_ylim(0, trans_churn['churn_rate'].max() + 12)

# Plot 5: Relationship Count vs Churn
colors5 = [RED if x <= 2 else GREEN for x in rel_churn['Total_Relationship_Count']]
bars5 = ax5.bar(rel_churn['Total_Relationship_Count'].astype(str),
                rel_churn['churn_rate'], color=colors5, width=0.6)
ax5.set_title('Churn Rate by Number of Products Held', fontweight='bold', color=NAVY, fontsize=11)
ax5.set_xlabel('Total Products / Relationships', fontsize=9)
ax5.set_ylabel('Churn Rate (%)', fontsize=9)
for bar, val in zip(bars5, rel_churn['churn_rate']):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val}%', ha='center', fontsize=9, fontweight='bold', color=NAVY)
ax5.set_ylim(0, rel_churn['churn_rate'].max() + 12)
red_p = mpatches.Patch(color=RED, label='High Churn Risk')
green_p = mpatches.Patch(color=GREEN, label='Lower Churn Risk')
ax5.legend(handles=[red_p, green_p], fontsize=8)

# Plot 6: Avg Transaction Amount — Churned vs Retained
ax6.set_facecolor('white')
labels = ['Retained', 'Churned']
vals = [avg_trans.get('Existing Customer', 0), avg_trans.get('Attrited Customer', 0)]
colors6 = [GREEN, RED]
bars6 = ax6.bar(labels, vals, color=colors6, width=0.4)
ax6.set_title('Avg Transaction Amount: Retained vs Churned', fontweight='bold', color=NAVY, fontsize=11)
ax6.set_ylabel('Avg Transaction Amount ($)', fontsize=9)
for bar, val in zip(bars6, vals):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'${val:,.0f}', ha='center', fontsize=11, fontweight='bold', color=NAVY)
ax6.set_ylim(0, max(vals) * 1.25)

# KPI Banner
kpi_ax = fig.add_axes([0.05, 0.915, 0.90, 0.042])
kpi_ax.set_facecolor(NAVY)
kpi_ax.set_xlim(0, 1)
kpi_ax.set_ylim(0, 1)
kpi_ax.axis('off')
kpis = [
    (f"{total:,}", "Total Customers"),
    (f"{churned:,}", "Churned Customers"),
    (f"{churn_rate}%", "Overall Churn Rate"),
    (f"{hr_total:,}", "High-Risk Customers"),
    (f"{hr_rate}%", "High-Risk Churn Rate"),
]
for i, (val, label) in enumerate(kpis):
    x = 0.1 + i * 0.20
    kpi_ax.text(x, 0.72, val, ha='center', fontsize=13, fontweight='bold', color='white')
    kpi_ax.text(x, 0.20, label, ha='center', fontsize=8, color=LIGHT)

# Recommendations Box
rec_ax = fig.add_axes([0.05, 0.06, 0.90, 0.165])
rec_ax.set_facecolor(NAVY)
rec_ax.set_xlim(0, 1)
rec_ax.set_ylim(0, 1)
rec_ax.axis('off')
rec_ax.text(0.5, 0.92, 'BUSINESS RECOMMENDATIONS', ha='center', fontsize=12,
            fontweight='bold', color='white')
recs = [
    ("1. Target Low-Transaction Customers",
     "Customers with fewer than 40 transactions show 60%+ churn. Trigger re-engagement campaigns with personalized offers."),
    ("2. Intervene at 3+ Inactive Months",
     "Churn spikes sharply after 3 months of inactivity. Automate outreach at the 2-month mark to prevent escalation."),
    ("3. Cross-Sell to Single-Product Holders",
     "Customers with 1-2 products churn at 2x the rate of those with 3+. Prioritize cross-sell initiatives for this segment."),
    ("4. Protect High-Income Segments",
     "While lower income shows higher churn, high-income churners represent greater revenue loss. Assign relationship managers proactively."),
]
for i, (title, desc) in enumerate(recs):
    x = 0.02 + (i % 2) * 0.50
    y = 0.65 if i < 2 else 0.25
    rec_ax.text(x, y, title, fontsize=9, fontweight='bold', color=LIGHT)
    rec_ax.text(x, y - 0.17, desc, fontsize=8, color='#CCCCCC',
                wrap=True, va='top')

fig.text(0.5, 0.025, 'Analysis by Sai Akshay Jilla  |  Tools: Python (Pandas, NumPy, Matplotlib, Seaborn)  |  Dataset: Credit Card Customers (Kaggle)',
         ha='center', fontsize=9, color='#888888')

plt.savefig('/mnt/user-data/outputs/Bank_Churn_Analysis.png', dpi=150, bbox_inches='tight',
            facecolor=GRAY)
plt.close()

print("=" * 60)
print("PROJECT 2: BANK CUSTOMER CHURN — KEY FINDINGS")
print("=" * 60)
print(f"\nTotal Customers: {total:,} | Churned: {churned:,} | Churn Rate: {churn_rate}%")
print(f"\nHigh-Risk Segment: {hr_total} customers | Churn Rate: {hr_rate}%")
print("\nAVG TRANSACTION — Retained vs Churned:")
print(avg_trans)
print("\nCHURN BY CARD CATEGORY:")
print(card_churn.to_string(index=False))
print("\nDone. Chart saved.")
