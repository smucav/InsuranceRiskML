# End-to-End Insurance Risk Analytics & Predictive Modeling

## Overview
This project analyzes historical car insurance data for AlphaCare Insurance Solutions (ACIS) to optimize marketing strategies and identify low-risk customer segments for premium adjustments. The analysis includes EDA, A/B hypothesis testing, and predictive modeling using machine learning.

## Setup Instructions
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Install Jupyter: `pip install jupyter`
4. Run scripts from `scripts/` or open notebooks in `notebooks/` with `jupyter notebook`.

## Project Structure
- `data/`: Raw and processed datasets.
- `notebooks/`: Jupyter notebooks for EDA and visualization.
- `scripts/`: Python modules for data loading and analysis.
- `.github/workflows/`: CI/CD configuration.

# Task 1: EDA Summary for Insurance Risk Analytics

## Overview
The Exploratory Data Analysis (EDA) on the insurance dataset (`MachineLearningRating_v3.txt`) reveals key insights into risk patterns for AlphaCare Insurance Solutions (ACIS). The dataset, with 837,833 cleaned rows and 47 columns, covers policy, client, vehicle, and claim details.

## Key Findings
1. **Loss Ratio by Province**:
   - Gauteng (0.282) and Limpopo (0.268) exhibit the highest loss ratios (claims/premiums), indicating higher risk. Northern Cape (0.116) is the lowest, suggesting potential for regional pricing adjustments.
   - Visualization: `plots/loss_ratio_province.png` (bar plot).
2. **Claims by Vehicle Type**:
   - Buses (loss ratio ~0.938) and Heavy Commercial vehicles (~0.571) have significantly higher claims than Passenger Vehicles (~0.221), highlighting elevated risk for commercial vehicles.
   - Visualization: `plots/claims_vehicletype.png` (box plot, log scale).
3. **Temporal Trends**:
   - Trends in average claims and premiums over time (2014-2015) may indicate seasonality or growth patterns, useful for marketing and risk management.
   - Visualization: `plots/temporal_trends.png` (line plot).
4. **Data Quality**:
   - Negative values in `totalpremium` and `totalclaims` were filtered.
   - High missingness in `gender` (~97%) limits its utility.
   - Dropped sparse columns (e.g., `customvalueestimate`) and imputed zeros in `totalpremium` using `calculatedpremiumperterm`.

## Implications
- **Pricing Strategy**: Adjust premiums in high-risk provinces (Gauteng, Limpopo) and for commercial vehicles (Buses, Heavy Commercial).
- **Marketing**: Target low-risk segments (e.g., Passenger Vehicles, Northern Cape) for customer acquisition.
- **Data Improvement**: Address negative values and missing `gender` data in future datasets.
