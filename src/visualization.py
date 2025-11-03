"""
Visualization module for tender analysis

This module creates visualizations for:
- Exploratory data analysis
- Model results
- Clustering patterns
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class TenderVisualizer:
    """Visualizer for tender data analysis"""
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent / "data" / "visualizations"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_award_distribution(self, df):
        """Plot distribution of award status"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        award_counts = df['award_status'].value_counts()
        axes[0].bar(award_counts.index, award_counts.values, color=['green', 'red'])
        axes[0].set_title('Distribution of Award Status')
        axes[0].set_xlabel('Status')
        axes[0].set_ylabel('Count')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Percentage
        award_pct = (award_counts / len(df) * 100).round(2)
        for i, (status, pct) in enumerate(award_pct.items()):
            axes[0].text(i, award_counts.iloc[i], f'{pct}%', 
                        ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(award_counts, labels=award_counts.index, autopct='%1.1f%%',
                   colors=['green', 'red'], startangle=90)
        axes[1].set_title('Award Status Proportion')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'award_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'award_distribution.png'}")
        plt.close()
    
    def plot_award_amounts(self, df):
        """Plot distribution of award amounts"""
        awarded_df = df[df['award_status'] == 'Awarded'].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(awarded_df['award_amount'], bins=50, color='skyblue', edgecolor='black')
        axes[0].set_title('Distribution of Award Amounts')
        axes[0].set_xlabel('Award Amount')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Log scale
        axes[1].hist(np.log1p(awarded_df['award_amount']), bins=50, 
                    color='lightcoral', edgecolor='black')
        axes[1].set_title('Distribution of Award Amounts (Log Scale)')
        axes[1].set_xlabel('Log(Award Amount + 1)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'award_amounts_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'award_amounts_distribution.png'}")
        plt.close()
    
    def plot_agency_analysis(self, df):
        """Plot agency-level analysis"""
        awarded_df = df[df['award_status'] == 'Awarded'].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Total spend by agency
        agency_spend = awarded_df.groupby('agency')['award_amount'].sum().sort_values(ascending=False)
        axes[0, 0].bar(range(len(agency_spend)), agency_spend.values, color='steelblue')
        axes[0, 0].set_xticks(range(len(agency_spend)))
        axes[0, 0].set_xticklabels(agency_spend.index, rotation=45)
        axes[0, 0].set_title('Total Spend by Agency')
        axes[0, 0].set_ylabel('Total Award Amount')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Number of awards by agency
        agency_count = df.groupby('agency').size().sort_values(ascending=False)
        axes[0, 1].bar(range(len(agency_count)), agency_count.values, color='orange')
        axes[0, 1].set_xticks(range(len(agency_count)))
        axes[0, 1].set_xticklabels(agency_count.index, rotation=45)
        axes[0, 1].set_title('Number of Tenders by Agency')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Average award by agency
        agency_avg = awarded_df.groupby('agency')['award_amount'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(agency_avg)), agency_avg.values, color='green')
        axes[1, 0].set_xticks(range(len(agency_avg)))
        axes[1, 0].set_xticklabels(agency_avg.index, rotation=45)
        axes[1, 0].set_title('Average Award Amount by Agency')
        axes[1, 0].set_ylabel('Average Award Amount')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Award rate by agency
        agency_award_rate = df.groupby('agency')['is_awarded'].mean().sort_values(ascending=False) * 100
        axes[1, 1].bar(range(len(agency_award_rate)), agency_award_rate.values, color='purple')
        axes[1, 1].set_xticks(range(len(agency_award_rate)))
        axes[1, 1].set_xticklabels(agency_award_rate.index, rotation=45)
        axes[1, 1].set_title('Award Rate by Agency (%)')
        axes[1, 1].set_ylabel('Award Rate (%)')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'agency_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'agency_analysis.png'}")
        plt.close()
    
    def plot_category_analysis(self, df):
        """Plot category-level analysis"""
        awarded_df = df[df['award_status'] == 'Awarded'].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Total spend by category
        cat_spend = awarded_df.groupby('category')['award_amount'].sum().sort_values(ascending=False)
        axes[0].barh(range(len(cat_spend)), cat_spend.values, color='teal')
        axes[0].set_yticks(range(len(cat_spend)))
        axes[0].set_yticklabels(cat_spend.index)
        axes[0].set_title('Total Spend by Category')
        axes[0].set_xlabel('Total Award Amount')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Average award by category
        cat_avg = awarded_df.groupby('category')['award_amount'].mean().sort_values(ascending=False)
        axes[1].barh(range(len(cat_avg)), cat_avg.values, color='coral')
        axes[1].set_yticks(range(len(cat_avg)))
        axes[1].set_yticklabels(cat_avg.index)
        axes[1].set_title('Average Award Amount by Category')
        axes[1].set_xlabel('Average Award Amount')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'category_analysis.png'}")
        plt.close()
    
    def plot_temporal_trends(self, df):
        """Plot temporal trends"""
        awarded_df = df[df['award_status'] == 'Awarded'].copy()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Awards over time
        monthly_awards = awarded_df.groupby([awarded_df['published_date'].dt.to_period('M')])['award_amount'].agg(['sum', 'count'])
        monthly_awards.index = monthly_awards.index.to_timestamp()
        
        axes[0].plot(monthly_awards.index, monthly_awards['count'], marker='o', color='blue', linewidth=2)
        axes[0].set_title('Number of Awards Over Time')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Number of Awards')
        axes[0].grid(alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Total spend over time
        axes[1].plot(monthly_awards.index, monthly_awards['sum'], marker='o', color='green', linewidth=2)
        axes[1].set_title('Total Award Amount Over Time')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Total Award Amount')
        axes[1].grid(alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'temporal_trends.png'}")
        plt.close()
    
    def plot_correlation_heatmap(self, df):
        """Plot correlation heatmap of numerical features"""
        # Select numerical columns
        num_cols = ['award_amount', 'published_year', 'published_month', 
                   'description_length', 'description_word_count', 'is_awarded',
                   'agency_encoded', 'category_encoded']
        
        # Filter to existing columns
        num_cols = [col for col in num_cols if col in df.columns]
        
        corr_matrix = df[num_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'correlation_heatmap.png'}")
        plt.close()
    
    def generate_all_visualizations(self, df):
        """Generate all visualizations"""
        print("\n=== Generating Visualizations ===")
        
        self.plot_award_distribution(df)
        self.plot_award_amounts(df)
        self.plot_agency_analysis(df)
        self.plot_category_analysis(df)
        self.plot_temporal_trends(df)
        self.plot_correlation_heatmap(df)
        
        print(f"\nAll visualizations saved to {self.output_dir}")


if __name__ == "__main__":
    # Load processed data
    data_path = Path(__file__).parent.parent / "data" / "tender_data_processed.csv"
    df = pd.read_csv(data_path)
    df['published_date'] = pd.to_datetime(df['published_date'])
    
    # Create visualizations
    visualizer = TenderVisualizer()
    visualizer.generate_all_visualizations(df)
