#!/usr/bin/env python3
"""
Comprehensive Analysis of Answer Positions in LongBench passage_retrieval_en

This script analyzes the positional distribution of answers in the passage retrieval task
to understand potential biases and patterns in long-context retrieval.

Author: Claude Code Analysis
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import seaborn as sns

class PassageRetrievalAnalyzer:
    """Analyzer for passage retrieval positional patterns"""
    
    def __init__(self, data_file: str):
        """Initialize with data file path"""
        self.data_file = data_file
        self.examples = []
        self.analysis_results = {}
        
    def load_data(self) -> None:
        """Load and parse the dataset"""
        print("📊 Loading passage_retrieval_en dataset...")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line.strip()) for line in f if line.strip()]
        
        print(f"✅ Loaded {len(self.examples)} examples")
        
    def extract_positional_data(self) -> List[Dict[str, Any]]:
        """Extract positional information for each example"""
        print("🔍 Extracting positional data...")
        
        positional_data = []
        
        for i, example in enumerate(self.examples):
            context = example['context']
            answer = example['answers'][0] if example['answers'] else None
            question = example['input']
            
            if not answer:
                continue
                
            # Extract paragraph number from answer
            paragraph_match = re.search(r'Paragraph (\d+)', answer)
            if not paragraph_match:
                continue
                
            paragraph_num = int(paragraph_match.group(1))
            
            # Find position in context
            answer_pos = context.find(answer)
            if answer_pos == -1:
                print(f"⚠️ Warning: Answer '{answer}' not found in context for example {i}")
                continue
            
            context_len = len(context)
            relative_pos = answer_pos / context_len if context_len > 0 else 0
            
            # Find all paragraph markers to understand structure
            paragraph_markers = re.findall(r'Paragraph \d+:', context)
            total_paragraphs = len(paragraph_markers)
            
            # Calculate relative paragraph position (1-based to 0-1 scale)
            relative_paragraph_pos = (paragraph_num - 1) / (total_paragraphs - 1) if total_paragraphs > 1 else 0
            
            # Extract additional features
            question_len = len(question)
            context_lines = context.count('\n') + 1
            
            # Find the actual paragraph content
            paragraph_start = context.find(f"Paragraph {paragraph_num}:")
            next_paragraph_match = re.search(rf'Paragraph {paragraph_num + 1}:', context)
            paragraph_end = next_paragraph_match.start() if next_paragraph_match else len(context)
            paragraph_content = context[paragraph_start:paragraph_end].strip()
            paragraph_content_len = len(paragraph_content)
            
            positional_data.append({
                'example_id': i,
                'paragraph_num': paragraph_num,
                'answer_char_pos': answer_pos,
                'relative_char_pos': relative_pos,
                'relative_paragraph_pos': relative_paragraph_pos,
                'context_len': context_len,
                'question_len': question_len,
                'total_paragraphs': total_paragraphs,
                'paragraph_content_len': paragraph_content_len,
                'context_lines': context_lines,
                'answer': answer,
                'question': question[:100] + '...' if len(question) > 100 else question
            })
        
        print(f"✅ Extracted positional data for {len(positional_data)} examples")
        return positional_data
    
    def analyze_position_distribution(self, positional_data: List[Dict]) -> Dict[str, Any]:
        """Analyze the distribution of answer positions"""
        print("📈 Analyzing position distributions...")
        
        # Extract position arrays
        paragraph_nums = [d['paragraph_num'] for d in positional_data]
        relative_char_positions = [d['relative_char_pos'] for d in positional_data]
        relative_paragraph_positions = [d['relative_paragraph_pos'] for d in positional_data]
        
        # Basic statistics
        stats = {
            'paragraph_numbers': {
                'mean': np.mean(paragraph_nums),
                'median': np.median(paragraph_nums),
                'std': np.std(paragraph_nums),
                'min': np.min(paragraph_nums),
                'max': np.max(paragraph_nums)
            },
            'relative_char_positions': {
                'mean': np.mean(relative_char_positions),
                'median': np.median(relative_char_positions),
                'std': np.std(relative_char_positions),
                'min': np.min(relative_char_positions),
                'max': np.max(relative_char_positions)
            },
            'relative_paragraph_positions': {
                'mean': np.mean(relative_paragraph_positions),
                'median': np.median(relative_paragraph_positions),
                'std': np.std(relative_paragraph_positions),
                'min': np.min(relative_paragraph_positions),
                'max': np.max(relative_paragraph_positions)
            }
        }
        
        # Distribution analysis
        paragraph_counter = Counter(paragraph_nums)
        
        # Position bias analysis
        position_bins = ['beginning', 'early_middle', 'late_middle', 'end']
        position_distribution = defaultdict(int)
        
        for pos in relative_char_positions:
            if pos < 0.25:
                position_distribution['beginning'] += 1
            elif pos < 0.5:
                position_distribution['early_middle'] += 1
            elif pos < 0.75:
                position_distribution['late_middle'] += 1
            else:
                position_distribution['end'] += 1
        
        return {
            'stats': stats,
            'paragraph_distribution': dict(paragraph_counter),
            'position_distribution': dict(position_distribution),
            'total_examples': len(positional_data)
        }
    
    def analyze_correlations(self, positional_data: List[Dict]) -> Dict[str, float]:
        """Analyze correlations between different variables"""
        print("🔗 Analyzing correlations...")
        
        # Prepare data for correlation analysis
        data_arrays = {
            'paragraph_num': [d['paragraph_num'] for d in positional_data],
            'relative_char_pos': [d['relative_char_pos'] for d in positional_data],
            'question_len': [d['question_len'] for d in positional_data],
            'context_len': [d['context_len'] for d in positional_data],
            'paragraph_content_len': [d['paragraph_content_len'] for d in positional_data]
        }
        
        correlations = {}
        variables = list(data_arrays.keys())
        
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                corr = np.corrcoef(data_arrays[var1], data_arrays[var2])[0, 1]
                correlations[f"{var1}_vs_{var2}"] = corr
        
        return correlations
    
    def generate_visualizations(self, positional_data: List[Dict], analysis: Dict) -> None:
        """Generate comprehensive visualizations"""
        print("📊 Generating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LongBench passage_retrieval_en: Positional Analysis', fontsize=16, fontweight='bold')
        
        # 1. Paragraph number distribution
        paragraph_nums = [d['paragraph_num'] for d in positional_data]
        axes[0, 0].hist(paragraph_nums, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Target Paragraph Numbers')
        axes[0, 0].set_xlabel('Paragraph Number')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_para = np.mean(paragraph_nums)
        median_para = np.median(paragraph_nums)
        axes[0, 0].axvline(mean_para, color='red', linestyle='--', label=f'Mean: {mean_para:.1f}')
        axes[0, 0].axvline(median_para, color='green', linestyle='--', label=f'Median: {median_para:.1f}')
        axes[0, 0].legend()
        
        # 2. Relative character position distribution
        relative_positions = [d['relative_char_pos'] for d in positional_data]
        axes[0, 1].hist(relative_positions, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Relative Character Positions')
        axes[0, 1].set_xlabel('Relative Position in Text (0.0 = start, 1.0 = end)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add uniform distribution line for comparison
        uniform_height = len(positional_data) / 20  # 20 bins
        axes[0, 1].axhline(uniform_height, color='red', linestyle='--', 
                          label=f'Uniform distribution: {uniform_height:.0f}')
        axes[0, 1].legend()
        
        # 3. Position bias (quartiles)
        position_dist = analysis['position_distribution']
        labels = list(position_dist.keys())
        values = list(position_dist.values())
        colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'gold']
        
        bars = axes[0, 2].bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 2].set_title('Position Bias Analysis (Text Quartiles)')
        axes[0, 2].set_ylabel('Number of Examples')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value}', ha='center', va='bottom')
        
        # 4. Scatter plot: Paragraph number vs Relative position
        paragraph_nums = [d['paragraph_num'] for d in positional_data]
        relative_positions = [d['relative_char_pos'] for d in positional_data]
        
        axes[1, 0].scatter(paragraph_nums, relative_positions, alpha=0.6, s=30)
        axes[1, 0].set_title('Paragraph Number vs Relative Character Position')
        axes[1, 0].set_xlabel('Paragraph Number')
        axes[1, 0].set_ylabel('Relative Character Position')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add trendline
        z = np.polyfit(paragraph_nums, relative_positions, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(paragraph_nums, p(paragraph_nums), "r--", alpha=0.8, 
                       label=f'Trendline (slope: {z[0]:.3f})')
        axes[1, 0].legend()
        
        # 5. Question length vs Answer position
        question_lens = [d['question_len'] for d in positional_data]
        axes[1, 1].scatter(question_lens, relative_positions, alpha=0.6, s=30)
        axes[1, 1].set_title('Question Length vs Answer Position')
        axes[1, 1].set_xlabel('Question Length (characters)')
        axes[1, 1].set_ylabel('Relative Character Position')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Heatmap of paragraph number frequency
        # Create a 6x5 grid representing paragraphs 1-30
        para_counts = Counter([d['paragraph_num'] for d in positional_data])
        heatmap_data = np.zeros((6, 5))
        
        for para_num, count in para_counts.items():
            row = (para_num - 1) // 5
            col = (para_num - 1) % 5
            if row < 6:  # Safety check
                heatmap_data[row, col] = count
        
        im = axes[1, 2].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        axes[1, 2].set_title('Paragraph Number Frequency Heatmap')
        
        # Add text annotations
        for i in range(6):
            for j in range(5):
                para_num = i * 5 + j + 1
                if para_num <= 30:
                    text = axes[1, 2].text(j, i, f'P{para_num}\\n{int(heatmap_data[i, j])}',
                                         ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'passage_retrieval_position_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to: {output_path}")
        
        plt.show()
    
    def generate_report(self, positional_data: List[Dict], analysis: Dict, correlations: Dict) -> None:
        """Generate a comprehensive text report"""
        print("📝 Generating comprehensive report...")
        
        report = []
        report.append("=" * 80)
        report.append("LONGBENCH PASSAGE_RETRIEVAL_EN: POSITIONAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset overview
        report.append("📊 DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total examples analyzed: {analysis['total_examples']}")
        report.append(f"Data file: {self.data_file}")
        report.append("")
        
        # Basic statistics
        report.append("📈 POSITIONAL STATISTICS")
        report.append("-" * 40)
        
        para_stats = analysis['stats']['paragraph_numbers']
        report.append(f"Target Paragraph Numbers:")
        report.append(f"  • Mean: {para_stats['mean']:.2f}")
        report.append(f"  • Median: {para_stats['median']:.1f}")
        report.append(f"  • Standard Deviation: {para_stats['std']:.2f}")
        report.append(f"  • Range: {para_stats['min']:.0f} - {para_stats['max']:.0f}")
        report.append("")
        
        char_stats = analysis['stats']['relative_char_positions']
        report.append(f"Relative Character Positions:")
        report.append(f"  • Mean: {char_stats['mean']:.3f} ({char_stats['mean']*100:.1f}% through text)")
        report.append(f"  • Median: {char_stats['median']:.3f} ({char_stats['median']*100:.1f}% through text)")
        report.append(f"  • Standard Deviation: {char_stats['std']:.3f}")
        report.append("")
        
        # Position bias analysis
        report.append("🎯 POSITION BIAS ANALYSIS")
        report.append("-" * 40)
        pos_dist = analysis['position_distribution']
        total = sum(pos_dist.values())
        
        report.append("Distribution across text quartiles:")
        for position, count in pos_dist.items():
            percentage = (count / total) * 100
            report.append(f"  • {position.replace('_', ' ').title()}: {count} examples ({percentage:.1f}%)")
        
        # Test for uniform distribution
        expected_per_quartile = total / 4
        chi_square_stat = sum((count - expected_per_quartile)**2 / expected_per_quartile 
                             for count in pos_dist.values())
        report.append(f"\\nChi-square test vs uniform distribution: {chi_square_stat:.2f}")
        report.append("(Higher values indicate more bias away from uniform distribution)")
        report.append("")
        
        # Most and least common paragraphs
        para_dist = analysis['paragraph_distribution']
        most_common = max(para_dist.items(), key=lambda x: x[1])
        least_common = min(para_dist.items(), key=lambda x: x[1])
        
        report.append("📊 PARAGRAPH FREQUENCY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Most frequent target: Paragraph {most_common[0]} ({most_common[1]} times)")
        report.append(f"Least frequent target: Paragraph {least_common[0]} ({least_common[1]} times)")
        
        # Count how many paragraphs appear with each frequency
        freq_counter = Counter(para_dist.values())
        report.append(f"\\nFrequency distribution:")
        for freq, count in sorted(freq_counter.items()):
            report.append(f"  • {count} paragraphs appear {freq} time(s)")
        report.append("")
        
        # Correlation analysis
        report.append("🔗 CORRELATION ANALYSIS")
        report.append("-" * 40)
        
        significant_correlations = [(k, v) for k, v in correlations.items() if abs(v) > 0.1]
        significant_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if significant_correlations:
            report.append("Notable correlations (|r| > 0.1):")
            for corr_name, corr_value in significant_correlations:
                var1, var2 = corr_name.replace('_vs_', ' vs ').replace('_', ' ').split(' vs ')
                report.append(f"  • {var1.title()} vs {var2.title()}: {corr_value:.3f}")
        else:
            report.append("No strong correlations found (all |r| < 0.1)")
        report.append("")
        
        # Key insights
        report.append("🔍 KEY INSIGHTS")
        report.append("-" * 40)
        
        # Position bias insight
        beginning_pct = (pos_dist['beginning'] / total) * 100
        end_pct = (pos_dist['end'] / total) * 100
        
        if beginning_pct > 30:
            report.append(f"• BEGINNING BIAS: {beginning_pct:.1f}% of answers in first quartile")
        elif end_pct > 30:
            report.append(f"• END BIAS: {end_pct:.1f}% of answers in last quartile")
        else:
            report.append("• RELATIVELY BALANCED: No strong positional bias detected")
        
        # Paragraph distribution insight
        para_range = para_stats['max'] - para_stats['min']
        para_coverage = para_range / 29  # 30 total paragraphs, so range is 0-29
        
        if para_coverage > 0.8:
            report.append(f"• GOOD COVERAGE: Targets span {para_range:.0f} of 30 paragraphs ({para_coverage*100:.0f}%)")
        else:
            report.append(f"• LIMITED COVERAGE: Targets only span {para_range:.0f} of 30 paragraphs ({para_coverage*100:.0f}%)")
        
        # Uniformity insight
        if chi_square_stat < 5:
            report.append("• UNIFORM DISTRIBUTION: Answer positions are relatively uniform")
        elif chi_square_stat < 15:
            report.append("• MODERATE BIAS: Some deviation from uniform distribution")
        else:
            report.append("• STRONG BIAS: Significant deviation from uniform distribution")
        
        report.append("")
        report.append("=" * 80)
        
        # Write report to file
        report_path = 'passage_retrieval_position_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(report))
        
        print(f"📝 Report saved to: {report_path}")
        
        # Also print to console
        print("\\n".join(report))
    
    def run_complete_analysis(self) -> None:
        """Run the complete positional analysis pipeline"""
        print("🚀 Starting comprehensive positional analysis...")
        print()
        
        # Load data
        self.load_data()
        
        # Extract positional information
        positional_data = self.extract_positional_data()
        
        if not positional_data:
            print("❌ No valid positional data found!")
            return
        
        # Analyze distributions
        analysis = self.analyze_position_distribution(positional_data)
        
        # Analyze correlations
        correlations = self.analyze_correlations(positional_data)
        
        # Generate visualizations
        self.generate_visualizations(positional_data, analysis)
        
        # Generate comprehensive report
        self.generate_report(positional_data, analysis, correlations)
        
        print()
        print("✅ Complete analysis finished!")
        print("📊 Check the generated visualization and report files.")


def main():
    """Main execution function"""
    data_file = 'test_data/real_benchmarks/longbench_passage_retrieval_en.jsonl'
    
    analyzer = PassageRetrievalAnalyzer(data_file)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()