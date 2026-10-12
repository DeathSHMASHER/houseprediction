import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11

def run_eda(data_path="data/housing.csv", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    print("=== DATASET OVERVIEW ===")
    print(df.info())
    print("\n=== SUMMARY STATISTICS ===")
    print(df.describe())
    print("\n=== MISSING VALUES COUNT ===")
    print(df.isnull().sum())
    
    # 1. Price Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Price'].dropna(), kde=True, color='#2563eb', bins=40, ax=ax)
    ax.set_title("Distribution of House Prices", fontsize=14, weight='bold', pad=12)
    ax.set_xlabel("Price ($)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "price_distribution.png")
    fig.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Area vs Price Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='Area', y='Price', hue='Location', alpha=0.75, palette='tab10', ax=ax)
    ax.set_title("Property Area vs. Price", fontsize=14, weight='bold', pad=12)
    ax.set_xlabel("Area (Square Feet)", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "area_vs_price.png")
    fig.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # 3. Bedrooms & Bathrooms vs Price (Boxplots)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x='Bedrooms', y='Price', palette='Blues', ax=axes[0])
    axes[0].set_title("Bedrooms vs Price", fontsize=13, weight='bold')
    axes[0].set_xlabel("Number of Bedrooms")
    axes[0].set_ylabel("Price ($)")

    sns.boxplot(data=df, x='Bathrooms', y='Price', palette='Purples', ax=axes[1])
    axes[1].set_title("Bathrooms vs Price", fontsize=13, weight='bold')
    axes[1].set_xlabel("Number of Bathrooms")
    axes[1].set_ylabel("Price ($)")
    plt.tight_layout()
    p3 = os.path.join(output_dir, "bedrooms_bathrooms_vs_price.png")
    fig.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")

    # 4. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=14, weight='bold', pad=12)
    plt.tight_layout()
    p4 = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")

if __name__ == '__main__':
    run_eda()
