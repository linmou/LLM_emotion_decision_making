import json
import numpy as np
from scipy import stats
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

def significant_sign(p_value: float) -> str:
    """
    Return significance marker based on p-value.
    
    Args:
        p_value: The p-value from statistical test
        
    Returns:
        '**' for p < 0.01 (highly significant)
        '*'  for p < 0.05 (significant)
    """
    if p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    return ''

def generate_text_description(results: Dict, category_a: Optional[str] = None, category_b: Optional[str] = None) -> str:
    """
    Generate a human-readable description of the statistical analysis results.
    
    Args:
        results: Dictionary containing either:
                - Single analysis results (emotion or intensity analysis)
                - Combined results with both emotion_analysis and intensity_analysis
        category_a: First category name (optional)
        category_b: Second category name (optional)
                
    Returns:
        A formatted string describing the results
    """
    # Check if this is a combined analysis (CSV case with both emotion and intensity)
    if 'emotion_analysis' in results and 'intensity_analysis' in results:
        descriptions = []
        descriptions.append("=== Emotion Analysis ===\n")
        descriptions.append(generate_text_description(results['emotion_analysis'], category_a, category_b))
        descriptions.append("\n\n=== Intensity Analysis ===")
        for emotion, intensity_results in results['intensity_analysis'].items():
            descriptions.append(f"\n\nResults for {str(emotion).capitalize()} Intensity Levels:")
            descriptions.append(generate_text_description(intensity_results, category_a, category_b))
        return "\n".join(descriptions)

    # Single analysis description (either emotion or intensity)
    description = []
    
    # Overall summary
    description.append("Statistical Analysis Summary:")
    description.append("=" * 30)
    
    # Individual conditions
    description.append("\n1. Individual Conditions:")
    for condition, data in results['individual_conditions'].items():
        # Handle both string and numeric conditions
        condition_str = (f"Intensity Level {condition}" 
                        if isinstance(condition, (int, float)) 
                        else str(condition).capitalize())
        description.append(f"\n{condition_str}:")
        
        # For each category, add a line with counts and percentages
        for i, (cat, ratio_key) in enumerate([(cat, f"{cat}_ratio") for cat in data['counts'].keys()]):
            count = data['counts'].get(cat, 0)
            ratio = data.get(ratio_key, 0)
            description.append(f"- {cat} decisions: {count} ({ratio*100:.1f}%)")
    
    # Overall test results
    description.append("\n2. Overall Statistical Test:")
    overall_test = results['overall_test']
    description.append(f"Chi-square value: {overall_test['chi_square']:.2f}")
    description.append(f"P-value: {overall_test['p_value']:.4f}")
    
    if overall_test['significant']:
        description.append("Conclusion: There are statistically significant differences in behavior across conditions (p < 0.05)")
    else:
        description.append("Conclusion: No statistically significant differences in behavior across conditions (p ≥ 0.05)")
    
    # Pairwise comparisons
    if results.get('pairwise_comparisons'):
        description.append("\n3. Pairwise Comparisons:")
        significant_pairs = []
        non_significant_pairs = []
        
        for comparison, stats in results['pairwise_comparisons'].items():
            conditions = comparison.replace('_vs_', ' vs ')
            if any(c.replace('.', '').isdigit() for c in conditions.split(' vs ')):
                conditions = f"Intensity {conditions}"
            
            sig_marker = f" {stats['significance_level']}" if stats['significance_level'] else ''
            result_str = f"- {conditions}{sig_marker} (p={stats['p_value']:.4f})"
            
            if stats['significant']:
                significant_pairs.append(result_str)
            else:
                non_significant_pairs.append(result_str)
        
        if significant_pairs:
            description.append("\nSignificant differences found between:")
            description.extend(significant_pairs)
        
        if non_significant_pairs:
            description.append("\nNo significant differences found between:")
            description.extend(non_significant_pairs)
    
    return "\n".join(description)

class BaseAnalyzer(ABC):
    """Base class for statistical analysis of behavioral data"""
    
    def __init__(self):
        super().__init__()
        self.categories = None
        
    def calculate_behavior_ratios(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Calculate the ratio of behaviors for all categories."""
        total = sum(counts.values())
        if total == 0:
            return {cat: 0.0 for cat in self.categories}
        return {cat: counts.get(cat, 0)/total for cat in self.categories}

    def chi_square_test(self, condition1_counts: Dict[str, int], 
                       condition2_counts: Dict[str, int]) -> Tuple[float, float]:
        """
        Perform chi-square test of independence between two conditions.
        Uses Fisher's exact test when counts are low or contain zeros.
        
        The contingency table will be:
        
               cat1  cat2  cat3  ...  catN
        cond1   x11   x12   x13  ...  x1N
        cond2   x21   x22   x23  ...  x2N
        
        Where xij is the count of category j in condition i
        """
        # Create contingency table for all categories
        contingency = np.array([
            [condition1_counts.get(cat, 0) for cat in self.categories],
            [condition2_counts.get(cat, 0) for cat in self.categories]
        ])
        
        try:
            # Use Fisher's exact test when we have zeros or low counts
            if np.any(contingency == 0) or np.any(contingency < 5):
                # Note: Fisher's exact test can handle RxC tables in scipy
                _, p_value = stats.fisher_exact(contingency)
                # Calculate Cramer's V for effect size
                n = np.sum(contingency)
                min_dim = min(contingency.shape) - 1  # min(rows-1, cols-1)
                v = np.sqrt(p_value * n / (n * min_dim))
                chi2 = v * n
                return chi2, p_value
            
            # Use chi-square test for larger counts
            # This naturally handles RxC contingency tables
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
            return chi2, p_value
        except ValueError:
            # Fallback in case of any other statistical issues
            return 0.0, 1.0

class BehaviorAnalyzer(BaseAnalyzer):
    """Unified analyzer for both JSON and CSV data formats"""
    
    def __init__(self):
        self.plotter = BehaviorVisualizer()

    def load_data(self, file_path: str) -> Union[List[Dict], pd.DataFrame]:
        """Load data from either JSON or CSV file"""
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            required_columns = ['emotion', 'intensity', 'category']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV missing required columns: {required_columns}")
            return df
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")

    def analyze_data(self, data_source: str, output_dir: Optional[str] = None) -> Dict:
        """
        Unified analysis method for both JSON and CSV data
        
        Args:
            data_source: Either a path to CSV file or dictionary of emotion JSON files
            output_dir: Directory to save analysis results
            
        Returns:
            Comprehensive analysis results
        """
        if output_dir is None:
            output_dir = Path(data_source).parent / 'plots'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if isinstance(data_source, str):
            df = self.load_data(data_source)
            self.categories = sorted(df.category.unique())
            self.plotter.update_category_labels(self.categories)
            
            emotion_results = self._analyze_emotion_effects(df)
            intensity_results = self._analyze_intensity_effects(df)
            self.plotter.plot_results(
                emotion_results, 
                f"{output_dir}/emotion_analysis.png",
                title="Emotion Analysis"
            )
            
            for emotion, results in intensity_results.items():
                self.plotter.plot_results(
                    results,
                    f"{output_dir}/{emotion}_intensity_analysis.png",
                    title=f"{emotion} Intensity Analysis"
                )
            
            full_results = {
                'emotion_analysis': emotion_results,
                'intensity_analysis': intensity_results
            }
            
        else:  # JSON case
            results = self._analyze_multiple_emotions(data_source)
            self.plotter.plot_results(
                results,
                f"{output_dir}/emotion_analysis.png",
                title="Emotion Analysis"
            )
            full_results = results

        return self.format_full_results(full_results)

    def format_full_results(self, analysis_results: Dict) -> Dict:
        """Add text descriptions and structure to results."""
        # Safe handling of categories for text description
        if len(self.categories) >= 2:
            category_a, category_b = self.categories[0], self.categories[1]
        elif len(self.categories) == 1:
            category_a, category_b = self.categories[0], None
        else:
            category_a, category_b = None, None
            
        return {
            **analysis_results,
            'text_description': generate_text_description(analysis_results, category_a, category_b)
        }

    def _analyze_emotion_effects(self, df: pd.DataFrame) -> Dict:
        """Analyze behavioral differences across emotions in CSV data"""
        emotion_counts = self._group_counts(df, 'emotion')
        return self._analyze_conditions(emotion_counts)

    def _analyze_intensity_effects(self, df: pd.DataFrame) -> Dict:
        """Analyze intensity effects within each emotion in CSV data"""
        results = {}
        for emotion in df['emotion'].unique():
            emotion_df = df[df['emotion'] == emotion]
            intensity_counts = self._group_counts(emotion_df, 'intensity')
            if len(intensity_counts) > 1:
                results[emotion] = self._analyze_conditions(intensity_counts)
        return results

    def _analyze_multiple_emotions(self, emotion_files: Dict[str, str]) -> Dict:
        """Analyze behavioral differences across multiple emotion JSON files"""
        emotion_data = {}
        for emotion, file_path in emotion_files.items():
            data = self.load_data(file_path)
            counts = self._extract_behavior_counts(data)
            ratios = self.calculate_behavior_ratios(counts)
            
            emotion_data[emotion] = {
                'counts': counts,
                **{f'{cat}_ratio': ratio for cat, ratio in ratios.items()}
            }
        
        return self._analyze_conditions(
            {emotion: data['counts'] for emotion, data in emotion_data.items()}
        )

    def _group_counts(self, df: pd.DataFrame, group_col: str) -> Dict[str, Dict]:
        """Calculate behavior counts for grouped data"""
        groups = df.groupby(group_col)
        return {
            group_name: group.category.value_counts().to_dict()
            for group_name, group in groups
        }

    def _extract_behavior_counts(self, data: List[Dict]) -> Dict[str, int]:
        """Extract counts from JSON data"""
        behavior_counts = {cat: 0 for cat in self.categories}
        for item in data:
            category = item.get('category', '').lower()
            behavior_counts[category] = behavior_counts.get(category, 0) + 1
            
        return behavior_counts

    def _analyze_conditions(self, condition_counts: Dict[str, Dict]) -> Dict:
        """Core analysis logic for any grouped conditions"""
        # Calculate ratios and prepare data
        individual_conditions = {}
        for condition, counts in condition_counts.items():
            ratios = self.calculate_behavior_ratios(counts)
            individual_conditions[condition] = {
                'counts': counts,
                **{f'{cat}_ratio': ratio for cat, ratio in ratios.items()}
            }

        # Perform overall statistical test
        overall_chi2 = 0.0
        overall_p_value = 1.0
        conditions = list(condition_counts.values())
        
        if len(conditions) >= 2:
            # Create contingency table for all categories
            # This creates an MxN table where:
            # M = number of conditions
            # N = number of categories
            contingency = np.array([
                [c.get(cat, 0) for cat in self.categories] 
                for c in conditions
            ])
            
            try:
                # Chi-square test naturally handles RxC tables
                overall_chi2, overall_p_value, _, _ = stats.chi2_contingency(contingency)
            except:
                pass

        # Perform pairwise comparisons
        pairwise_comparisons = {}
        for (cond1, cond2) in combinations(condition_counts.keys(), 2):
            chi2, p_value = self.chi_square_test(
                condition_counts[cond1],
                condition_counts[cond2]
            )
            pairwise_comparisons[f"{cond1}_vs_{cond2}"] = {
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'significance_level': significant_sign(p_value)
            }

        return {
            'individual_conditions': individual_conditions,
            'overall_test': {
                'chi_square': float(overall_chi2),
                'p_value': float(overall_p_value),
                'significant': bool(overall_p_value < 0.05)
            },
            'pairwise_comparisons': pairwise_comparisons
        }

class BehaviorVisualizer:
    """Unified visualization class"""
    
    def update_category_labels(self, categories: List[str]):
        self.categories = categories
        cmap = plt.colormaps['tab20']
        self.colors = cmap(np.linspace(0, 1, len(categories)))
    
    def plot_results(self, results: Dict, output_path: str, title: str = None) -> None:
        """
        Unified plotting method for all types of results
        
        Args:
            results: Analysis results dictionary
            output_path: Path to save the plot
            title: Optional title for the plot
        """
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(15, 10))
        
        if len(self.categories) >= 2:
            # Use original layout with 2x2 grid for multiple categories
            gs = fig.add_gridspec(2, 2)
            self._plot_behavior_rates(results, fig.add_subplot(gs[0, 0]))
            self._plot_pvalue_heatmap(results, fig.add_subplot(gs[0, 1]))
            self._plot_raw_counts(results, fig.add_subplot(gs[1, :]))
        else:
            # Simplified layout for single category
            gs = fig.add_gridspec(2, 1)
            self._plot_single_category_rates(results, fig.add_subplot(gs[0, 0]))
            self._plot_pvalue_heatmap(results, fig.add_subplot(gs[1, 0]))
        
        if title:
            fig.suptitle(title, fontsize=16, y=1.02)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_behavior_rates(self, results: Dict, ax: plt.Axes) -> None:
        """Plot behavior rates with percentages for multiple categories"""
        # Skip if there's only one category - this will be handled by _plot_single_category_rates
        if len(self.categories) < 2:
            return
            
        conditions = list(results['individual_conditions'].keys())
        n_conditions = len(conditions)
        n_categories = len(self.categories)
        
        width = 0.8 / n_categories
        x = np.arange(n_conditions)
        
        for i, (category, color) in enumerate(zip(self.categories, self.colors)):
            rates = [results['individual_conditions'][cond][f'{category}_ratio'] 
                    for cond in conditions]
            pos = x + (i - (n_categories-1)/2) * width
            bars = ax.bar(pos, rates, width, label=category, color=color, alpha=0.6)
            
            # Add percentage labels
            for j, rate in enumerate(rates):
                ax.text(pos[j], rate, f'{rate*100:.1f}%', 
                       ha='center', va='bottom')
        
        ax.set_ylabel('Ratio')
        ax.set_title('Behavior Ratios by Condition')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        ax.set_ylim(0, 1.2)

    def _plot_pvalue_heatmap(self, results: Dict, ax: plt.Axes) -> None:
        """Plot p-value heatmap for both emotion and intensity comparisons"""
        # Get list of conditions from the results
        conditions = list(results['individual_conditions'].keys())
        n_conditions = len(conditions)
        p_values_matrix = np.ones((n_conditions, n_conditions))
        
        # Convert conditions to strings for consistent handling
        condition_labels = [str(cond) for cond in conditions]
        
        # Fill the p-values matrix
        for pair, stats in results['pairwise_comparisons'].items():
            # Split on _vs_ and handle potential numeric values
            cond1, cond2 = pair.split('_vs_')
            # Convert to string to match our condition_labels
            i = condition_labels.index(str(cond1))
            j = condition_labels.index(str(cond2))
            p_values_matrix[i, j] = stats['p_value']
            p_values_matrix[j, i] = stats['p_value']  # Mirror the matrix
        
        # Create heatmap with formatted labels
        sns.heatmap(
            p_values_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=condition_labels,
            yticklabels=condition_labels,
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Set title based on whether we're dealing with emotions or intensities
        if any(isinstance(cond, (int, float)) for cond in conditions):
            ax.set_title('P-values Heatmap for Intensity Level Comparisons')
        else:
            ax.set_title('P-values Heatmap for Emotion Comparisons')

    def _plot_single_category_rates(self, results: Dict, ax: plt.Axes) -> None:
        """Plot behavior rates for a single category"""
        conditions = list(results['individual_conditions'].keys())
        n_conditions = len(conditions)
        category = self.categories[0]  # We know there's only one
        
        rates = [results['individual_conditions'][cond][f'{category}_ratio'] 
                for cond in conditions]
        
        x = np.arange(n_conditions)
        bars = ax.bar(x, rates, 0.6, label=category, color=self.colors[0], alpha=0.6)
        
        # Add percentage labels
        for j, rate in enumerate(rates):
            ax.text(x[j], rate, f'{rate*100:.1f}%', 
                   ha='center', va='bottom')
        
        ax.set_ylabel('Ratio')
        ax.set_title(f'{category} Ratios by Condition')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        ax.set_ylim(0, 1.2)

    def _plot_raw_counts(self, results: Dict, ax: plt.Axes) -> None:
        """Plot behavior ratios"""
        conditions = list(results['individual_conditions'].keys())
        
        # Handle the case of only one category
        if len(self.categories) == 1:
            single_category = self.categories[0]
            ratios = [results['individual_conditions'][cond][f'{single_category}_ratio'] 
                      for cond in conditions]
            
            x = np.arange(len(conditions))
            ax.bar(x, ratios, 0.6, label=single_category, color='green', alpha=0.6)
            
            # Add percentage labels
            for i, ratio in enumerate(ratios):
                ax.text(i, ratio, f'{ratio*100:.1f}%', 
                       ha='center', va='bottom')
                
            ax.set_ylabel('Ratio')
            ax.set_title('Behavior Ratios by Condition')
            ax.set_xticks(x)
            ax.set_xticklabels(conditions)
            ax.legend()
            ax.set_ylim(0, 1.2)
            return
        
        # Original implementation for 2+ categories
        coop_ratios = []
        defect_ratios = []
        
        for condition in conditions:
            data = results['individual_conditions'][condition]
            coop_ratios.append(data[f'{self.categories[0]}_ratio'])
            defect_ratios.append(data[f'{self.categories[1]}_ratio'])
        
        x = np.arange(len(conditions))
        width = 0.35
        
        ax.bar(x - width/2, coop_ratios, width, label=self.categories[0], color='green', alpha=0.6)
        ax.bar(x + width/2, defect_ratios, width, label=self.categories[1], color='red', alpha=0.6)
        ax.set_ylabel('Ratio')
        ax.set_title('Behavior Ratios by Condition')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        
        # Add percentage labels on top of each bar
        for i, (coop, defect) in enumerate(zip(coop_ratios, defect_ratios)):
            ax.text(i - width/2, coop, f'{coop*100:.1f}%', 
                   ha='center', va='bottom')
            ax.text(i + width/2, defect, f'{defect*100:.1f}%', 
                   ha='center', va='bottom')
        
        # Set y-axis limits to accommodate labels
        ax.set_ylim(0, 1.2)

def analyze_emotion_and_intensity_effects(csv_file_path: str, output_dir: Optional[str] = None) -> Dict:
    analyzer = BehaviorAnalyzer()
    return analyzer.analyze_data(csv_file_path)

# Simplified usage
if __name__ == "__main__":
    analyzer = BehaviorAnalyzer()
    
    # Example with CSV
    csv_results = analyzer.analyze_data(
       'results/trust_game_trustor_20250210_164334/all_output_samples.csv' 
    )

    with open('results/trust_game_trustor_20250210_164334/analysis_results.json', 'w') as f:
        json.dump(csv_results, f, indent=4)