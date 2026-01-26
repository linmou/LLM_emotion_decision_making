#!/usr/bin/env python3
"""
Comprehensive Multi-Task Position Analysis for Long-Context Benchmarks

This script analyzes positional distribution of answers across three high-priority tasks:
1. LongBench Passage Retrieval (discrete paragraph positions)
2. InfiniteBench PassKey (needle-in-haystack character positions) 
3. InfiniteBench KV-Retrieval (structured JSON key positions)

Author: Claude Code Analysis
Date: August 2025
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional
import seaborn as sns
from pathlib import Path
import uuid
from abc import ABC, abstractmethod

class BaseTaskAnalyzer(ABC):
    """Abstract base class for task-specific position analyzers"""
    
    def __init__(self, data_file: str, task_name: str):
        self.data_file = data_file
        self.task_name = task_name
        self.examples = []
        
    def load_data(self) -> None:
        """Load and parse the dataset"""
        print(f"📊 Loading {self.task_name} dataset...")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line.strip()) for line in f if line.strip()]
        
        print(f"✅ Loaded {len(self.examples)} examples")
    
    @abstractmethod
    def extract_positional_data(self) -> List[Dict[str, Any]]:
        """Extract positional information - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_task_specific_metrics(self, positional_data: List[Dict]) -> Dict[str, Any]:
        """Get task-specific analysis metrics"""
        pass

class PassageRetrievalAnalyzer(BaseTaskAnalyzer):
    """Analyzer for LongBench passage retrieval tasks"""
    
    def __init__(self, data_file: str):
        super().__init__(data_file, "Passage Retrieval")
        
    def extract_positional_data(self) -> List[Dict[str, Any]]:
        """Extract positional information for passage retrieval task"""
        print("🔍 Extracting passage retrieval positional data...")
        
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
            
            # Find all paragraph markers
            paragraph_markers = re.findall(r'Paragraph \d+:', context)
            total_paragraphs = len(paragraph_markers)
            
            # Calculate relative paragraph position (1-based to 0-1 scale)
            relative_paragraph_pos = (paragraph_num - 1) / (total_paragraphs - 1) if total_paragraphs > 1 else 0
            
            # Find the actual paragraph content
            paragraph_start = context.find(f"Paragraph {paragraph_num}:")
            next_paragraph_match = re.search(rf'Paragraph {paragraph_num + 1}:', context)
            paragraph_end = next_paragraph_match.start() if next_paragraph_match else len(context)
            paragraph_content = context[paragraph_start:paragraph_end].strip()
            
            positional_data.append({
                'task_type': 'passage_retrieval',
                'example_id': i,
                'paragraph_num': paragraph_num,
                'answer_char_pos': answer_pos,
                'relative_char_pos': relative_pos,
                'relative_paragraph_pos': relative_paragraph_pos,
                'context_len': context_len,
                'total_paragraphs': total_paragraphs,
                'paragraph_content_len': len(paragraph_content),
                'answer': answer,
                'question': question[:100] + '...' if len(question) > 100 else question
            })
        
        print(f"✅ Extracted positional data for {len(positional_data)} examples")
        return positional_data
    
    def get_task_specific_metrics(self, positional_data: List[Dict]) -> Dict[str, Any]:
        """Get passage retrieval specific metrics"""
        paragraph_nums = [d['paragraph_num'] for d in positional_data]
        total_paragraphs_list = [d['total_paragraphs'] for d in positional_data]
        
        return {
            'paragraph_stats': {
                'mean': np.mean(paragraph_nums),
                'median': np.median(paragraph_nums),
                'std': np.std(paragraph_nums),
                'min': np.min(paragraph_nums),
                'max': np.max(paragraph_nums)
            },
            'paragraph_distribution': dict(Counter(paragraph_nums)),
            'avg_total_paragraphs': np.mean(total_paragraphs_list)
        }

class PassKeyAnalyzer(BaseTaskAnalyzer):
    """Analyzer for InfiniteBench passkey retrieval tasks"""
    
    def __init__(self, data_file: str):
        super().__init__(data_file, "PassKey Retrieval")
        
    def extract_positional_data(self) -> List[Dict[str, Any]]:
        """Extract positional information for passkey task"""
        print("🔍 Extracting passkey positional data...")
        
        positional_data = []
        
        for i, example in enumerate(self.examples):
            context = example['context']
            # InfiniteBench uses 'answer' field - can be string or list
            answer = example.get('answer', '')
            if isinstance(answer, list):
                answer = answer[0] if answer else ''
            question = example.get('input', example.get('question', ''))
            
            if not answer:
                continue
            
            # Find all occurrences of the answer (passkey) in context
            answer_positions = []
            start = 0
            while True:
                pos = context.find(answer, start)
                if pos == -1:
                    break
                answer_positions.append(pos)
                start = pos + 1
            
            if not answer_positions:
                print(f"⚠️ Warning: PassKey '{answer}' not found in context for example {i}")
                continue
            
            # Use the first occurrence for position analysis
            first_answer_pos = answer_positions[0]
            context_len = len(context)
            relative_pos = first_answer_pos / context_len if context_len > 0 else 0
            
            # Analyze context structure (repetitive patterns)
            lines = context.split('\n')
            repetitive_pattern_count = len([line for line in lines if 'grass is green' in line.lower()])
            
            positional_data.append({
                'task_type': 'passkey',
                'example_id': i,
                'answer_char_pos': first_answer_pos,
                'relative_char_pos': relative_pos,
                'context_len': context_len,
                'answer_occurrences': len(answer_positions),
                'all_answer_positions': answer_positions,
                'repetitive_pattern_count': repetitive_pattern_count,
                'context_lines': len(lines),
                'answer': answer,
                'question': question[:100] + '...' if len(question) > 100 else question
            })
        
        print(f"✅ Extracted positional data for {len(positional_data)} examples")
        return positional_data
    
    def get_task_specific_metrics(self, positional_data: List[Dict]) -> Dict[str, Any]:
        """Get passkey specific metrics"""
        context_lens = [d['context_len'] for d in positional_data]
        answer_occurrences = [d['answer_occurrences'] for d in positional_data]
        
        return {
            'context_length_stats': {
                'mean': np.mean(context_lens),
                'median': np.median(context_lens),
                'std': np.std(context_lens),
                'min': np.min(context_lens),
                'max': np.max(context_lens)
            },
            'answer_occurrence_stats': {
                'mean': np.mean(answer_occurrences),
                'median': np.median(answer_occurrences),
                'max': np.max(answer_occurrences)
            },
            'avg_context_length': np.mean(context_lens)
        }

class KVRetrievalAnalyzer(BaseTaskAnalyzer):
    """Analyzer for InfiniteBench key-value retrieval tasks"""
    
    def __init__(self, data_file: str):
        super().__init__(data_file, "KV Retrieval")
        
    def extract_positional_data(self) -> List[Dict[str, Any]]:
        """Extract positional information for KV retrieval task"""
        print("🔍 Extracting KV retrieval positional data...")
        
        positional_data = []
        
        for i, example in enumerate(self.examples):
            context = example['context']
            answer = example.get('answer', '')
            if isinstance(answer, list):
                answer = answer[0] if answer else ''
            question = example.get('input', example.get('question', ''))
            
            if not answer:
                continue
                
            # Extract JSON data from context
            json_start = context.find('{')
            json_end = context.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print(f"⚠️ Warning: No JSON found in context for example {i}")
                continue
                
            json_str = context[json_start:json_end]
            
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Invalid JSON in context for example {i}")
                continue
            
            # Find the key that maps to our answer
            target_key = None
            for key, value in json_data.items():
                if value == answer:
                    target_key = key
                    break
            
            if target_key is None:
                print(f"⚠️ Warning: Answer '{answer}' not found as value in JSON for example {i}")
                continue
            
            # Calculate positions
            key_position_in_json = json_str.find(f'"{target_key}"')
            key_position_in_context = context.find(f'"{target_key}"')
            
            context_len = len(context)
            json_len = len(json_str)
            
            # Relative positions
            relative_key_pos_context = key_position_in_context / context_len if context_len > 0 else 0
            relative_key_pos_json = key_position_in_json / json_len if json_len > 0 else 0
            
            # JSON structure analysis
            total_keys = len(json_data)
            key_index = list(json_data.keys()).index(target_key)
            relative_key_index = key_index / (total_keys - 1) if total_keys > 1 else 0
            
            positional_data.append({
                'task_type': 'kv_retrieval',
                'example_id': i,
                'target_key': target_key,
                'key_char_pos_context': key_position_in_context,
                'key_char_pos_json': key_position_in_json,
                'relative_key_pos_context': relative_key_pos_context,
                'relative_key_pos_json': relative_key_pos_json,
                'key_index': key_index,
                'relative_key_index': relative_key_index,
                'context_len': context_len,
                'json_len': json_len,
                'total_keys': total_keys,
                'answer': answer,
                'question': question[:100] + '...' if len(question) > 100 else question
            })
        
        print(f"✅ Extracted positional data for {len(positional_data)} examples")
        return positional_data
    
    def get_task_specific_metrics(self, positional_data: List[Dict]) -> Dict[str, Any]:
        """Get KV retrieval specific metrics"""
        total_keys = [d['total_keys'] for d in positional_data]
        key_indices = [d['key_index'] for d in positional_data]
        json_lens = [d['json_len'] for d in positional_data]
        
        return {
            'json_structure_stats': {
                'avg_total_keys': np.mean(total_keys),
                'avg_json_length': np.mean(json_lens),
                'key_index_mean': np.mean(key_indices),
                'key_index_std': np.std(key_indices)
            },
            'key_distribution': dict(Counter(key_indices))
        }

class MultiTaskPositionAnalyzer:
    """Unified analyzer for multiple long-context tasks"""
    
    def __init__(self):
        self.analyzers = {}
        self.results = {}
        
    def add_task(self, task_name: str, analyzer: BaseTaskAnalyzer) -> None:
        """Add a task analyzer to the multi-task analysis"""
        self.analyzers[task_name] = analyzer
        
    def run_analysis(self) -> Dict[str, Any]:
        """Run analysis for all added tasks"""
        print("🚀 Starting multi-task position analysis...")
        print("=" * 80)
        
        all_results = {}
        all_positional_data = []
        
        for task_name, analyzer in self.analyzers.items():
            print(f"\n📋 Analyzing {task_name}...")
            print("-" * 50)
            
            # Load data and extract positions
            analyzer.load_data()
            positional_data = analyzer.extract_positional_data()
            
            if not positional_data:
                print(f"❌ No valid positional data found for {task_name}!")
                continue
            
            # Analyze positions
            task_results = self._analyze_single_task(positional_data, analyzer)
            all_results[task_name] = task_results
            
            # Add task type to positional data for cross-task analysis
            for item in positional_data:
                item['task_name'] = task_name
            all_positional_data.extend(positional_data)
        
        # Cross-task analysis
        if len(all_positional_data) > 0:
            cross_task_results = self._analyze_cross_task_patterns(all_positional_data)
            all_results['cross_task_analysis'] = cross_task_results
        
        # Generate comprehensive report
        self._generate_comprehensive_report(all_results, all_positional_data)
        
        # Generate visualizations
        self._generate_multi_task_visualizations(all_results, all_positional_data)
        
        print("\n✅ Multi-task analysis completed!")
        return all_results
    
    def _analyze_single_task(self, positional_data: List[Dict], analyzer: BaseTaskAnalyzer) -> Dict[str, Any]:
        """Analyze position distribution for a single task"""
        
        # Common position analysis
        relative_positions = [d.get('relative_char_pos', d.get('relative_key_pos_context', 0)) 
                            for d in positional_data]
        context_lens = [d.get('context_len', 0) for d in positional_data]
        
        # Position distribution (quartiles)
        position_distribution = defaultdict(int)
        for pos in relative_positions:
            if pos < 0.25:
                position_distribution['beginning'] += 1
            elif pos < 0.5:
                position_distribution['early_middle'] += 1
            elif pos < 0.75:
                position_distribution['late_middle'] += 1
            else:
                position_distribution['end'] += 1
        
        # Basic statistics
        stats = {
            'relative_positions': {
                'mean': np.mean(relative_positions),
                'median': np.median(relative_positions),
                'std': np.std(relative_positions),
                'min': np.min(relative_positions),
                'max': np.max(relative_positions)
            },
            'context_lengths': {
                'mean': np.mean(context_lens),
                'median': np.median(context_lens),
                'std': np.std(context_lens),
                'min': np.min(context_lens),
                'max': np.max(context_lens)
            }
        }
        
        # Chi-square test for uniformity
        total = sum(position_distribution.values())
        expected_per_quartile = total / 4
        chi_square_stat = sum((count - expected_per_quartile)**2 / expected_per_quartile 
                             for count in position_distribution.values())
        
        # Task-specific metrics
        task_specific_metrics = analyzer.get_task_specific_metrics(positional_data)
        
        return {
            'stats': stats,
            'position_distribution': dict(position_distribution),
            'chi_square_uniformity': chi_square_stat,
            'task_specific_metrics': task_specific_metrics,
            'total_examples': len(positional_data)
        }
    
    def _analyze_cross_task_patterns(self, all_positional_data: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns across different tasks"""
        print("🔍 Analyzing cross-task position patterns...")
        
        # Group by task type
        task_groups = defaultdict(list)
        for item in all_positional_data:
            task_groups[item['task_name']].append(item)
        
        # Compare position distributions
        cross_task_comparison = {}
        
        for task_name, task_data in task_groups.items():
            relative_positions = [d.get('relative_char_pos', d.get('relative_key_pos_context', 0)) 
                                for d in task_data]
            
            # Position bias metrics
            beginning_bias = sum(1 for pos in relative_positions if pos < 0.25) / len(relative_positions)
            middle_bias = sum(1 for pos in relative_positions if 0.25 <= pos < 0.75) / len(relative_positions)
            end_bias = sum(1 for pos in relative_positions if pos >= 0.75) / len(relative_positions)
            
            cross_task_comparison[task_name] = {
                'beginning_bias': beginning_bias,
                'middle_bias': middle_bias, 
                'end_bias': end_bias,
                'mean_position': np.mean(relative_positions),
                'position_std': np.std(relative_positions)
            }
        
        return cross_task_comparison
    
    def _generate_multi_task_visualizations(self, results: Dict, all_positional_data: List[Dict]) -> None:
        """Generate comprehensive visualizations for all tasks"""
        print("📊 Generating multi-task visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(3, 3, figsize=(21, 18))
        fig.suptitle('Multi-Task Long-Context Position Analysis', fontsize=18, fontweight='bold')
        
        # Task-specific data
        task_groups = defaultdict(list)
        for item in all_positional_data:
            task_groups[item['task_name']].append(item)
        
        colors = ['lightcoral', 'lightskyblue', 'lightgreen']
        task_names = list(task_groups.keys())
        
        # 1. Position distribution comparison across tasks
        for i, (task_name, task_data) in enumerate(task_groups.items()):
            relative_positions = [d.get('relative_char_pos', d.get('relative_key_pos_context', 0)) 
                                for d in task_data]
            
            axes[0, i].hist(relative_positions, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
            axes[0, i].set_title(f'{task_name}: Position Distribution')
            axes[0, i].set_xlabel('Relative Position')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)
            
            # Add uniform line
            uniform_height = len(relative_positions) / 20
            axes[0, i].axhline(uniform_height, color='red', linestyle='--', alpha=0.7,
                              label=f'Uniform: {uniform_height:.1f}')
            axes[0, i].legend()
        
        # 2. Position bias comparison (quartiles)
        quartile_data = {}
        for task_name, task_data in task_groups.items():
            relative_positions = [d.get('relative_char_pos', d.get('relative_key_pos_context', 0)) 
                                for d in task_data]
            
            quartile_counts = [
                sum(1 for pos in relative_positions if pos < 0.25),
                sum(1 for pos in relative_positions if 0.25 <= pos < 0.5),
                sum(1 for pos in relative_positions if 0.5 <= pos < 0.75),
                sum(1 for pos in relative_positions if pos >= 0.75)
            ]
            quartile_data[task_name] = [count / len(relative_positions) * 100 for count in quartile_counts]
        
        quartile_labels = ['Beginning', 'Early Mid', 'Late Mid', 'End']
        x = np.arange(len(quartile_labels))
        width = 0.25
        
        for i, (task_name, percentages) in enumerate(quartile_data.items()):
            axes[1, 0].bar(x + i * width, percentages, width, label=task_name, 
                          color=colors[i], alpha=0.8)
        
        axes[1, 0].set_title('Position Bias Comparison Across Tasks')
        axes[1, 0].set_xlabel('Position Quartile')
        axes[1, 0].set_ylabel('Percentage of Answers')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(quartile_labels)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3. Context length vs position scatter
        for i, (task_name, task_data) in enumerate(task_groups.items()):
            context_lens = [d.get('context_len', 0) for d in task_data]
            relative_positions = [d.get('relative_char_pos', d.get('relative_key_pos_context', 0)) 
                                for d in task_data]
            
            axes[1, 1].scatter(context_lens, relative_positions, alpha=0.6, s=30, 
                              label=task_name, color=colors[i])
        
        axes[1, 1].set_title('Context Length vs Answer Position')
        axes[1, 1].set_xlabel('Context Length (characters)')
        axes[1, 1].set_ylabel('Relative Position')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 4. Chi-square uniformity test comparison
        task_names_clean = []
        chi_square_values = []
        
        for task_name in task_names:
            if task_name in results and 'chi_square_uniformity' in results[task_name]:
                task_names_clean.append(task_name)
                chi_square_values.append(results[task_name]['chi_square_uniformity'])
        
        bars = axes[1, 2].bar(task_names_clean, chi_square_values, color=colors[:len(chi_square_values)], alpha=0.8)
        axes[1, 2].set_title('Position Uniformity (Chi-Square Test)')
        axes[1, 2].set_ylabel('Chi-Square Statistic')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, chi_square_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 5-9. Task-specific visualizations
        task_specific_plots = [
            (2, 0), (2, 1), (2, 2)
        ]
        
        for i, (task_name, task_data) in enumerate(task_groups.items()):
            if i >= 3:  # Only show first 3 tasks
                break
                
            row, col = task_specific_plots[i]
            
            if task_name == "Passage Retrieval":
                # Paragraph number distribution
                paragraph_nums = [d.get('paragraph_num', 0) for d in task_data if 'paragraph_num' in d]
                if paragraph_nums:
                    axes[row, col].hist(paragraph_nums, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
                    axes[row, col].set_title(f'{task_name}: Paragraph Distribution')
                    axes[row, col].set_xlabel('Paragraph Number')
                    axes[row, col].set_ylabel('Frequency')
            
            elif task_name == "PassKey Retrieval":
                # Context length distribution
                context_lens = [d.get('context_len', 0) for d in task_data]
                axes[row, col].hist(context_lens, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
                axes[row, col].set_title(f'{task_name}: Context Length Distribution')
                axes[row, col].set_xlabel('Context Length (characters)')
                axes[row, col].set_ylabel('Frequency')
            
            elif task_name == "KV Retrieval":
                # Key index distribution
                key_indices = [d.get('key_index', 0) for d in task_data if 'key_index' in d]
                if key_indices:
                    axes[row, col].hist(key_indices, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
                    axes[row, col].set_title(f'{task_name}: Key Index Distribution')
                    axes[row, col].set_xlabel('Key Index in JSON')
                    axes[row, col].set_ylabel('Frequency')
            
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'multi_task_position_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Multi-task visualizations saved to: {output_path}")
        
        plt.show()
    
    def _generate_comprehensive_report(self, results: Dict, all_positional_data: List[Dict]) -> None:
        """Generate comprehensive report for all tasks"""
        print("📝 Generating comprehensive multi-task report...")
        
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE MULTI-TASK LONG-CONTEXT POSITION ANALYSIS REPORT")
        report.append("=" * 100)
        report.append("")
        
        # Executive Summary
        report.append("📋 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        total_examples = len(all_positional_data)
        tasks_analyzed = len(self.analyzers)
        report.append(f"Total examples analyzed: {total_examples}")
        report.append(f"Tasks analyzed: {tasks_analyzed}")
        report.append(f"Analysis date: August 2025")
        report.append("")
        
        # Individual task results
        for task_name, task_results in results.items():
            if task_name == 'cross_task_analysis':
                continue
                
            report.append(f"🎯 {task_name.upper()} ANALYSIS")
            report.append("-" * 50)
            
            if 'total_examples' in task_results:
                report.append(f"Examples: {task_results['total_examples']}")
            
            if 'stats' in task_results:
                pos_stats = task_results['stats']['relative_positions']
                report.append(f"Mean position: {pos_stats['mean']:.3f} ({pos_stats['mean']*100:.1f}% through text)")
                report.append(f"Position std dev: {pos_stats['std']:.3f}")
            
            if 'position_distribution' in task_results:
                pos_dist = task_results['position_distribution']
                total = sum(pos_dist.values())
                report.append("Position distribution:")
                for position, count in pos_dist.items():
                    percentage = (count / total) * 100
                    report.append(f"  • {position.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            
            if 'chi_square_uniformity' in task_results:
                chi_sq = task_results['chi_square_uniformity']
                report.append(f"Chi-square uniformity test: {chi_sq:.2f}")
                if chi_sq < 5:
                    report.append("  → UNIFORM: Positions are relatively uniform")
                elif chi_sq < 15:
                    report.append("  → MODERATE BIAS: Some deviation from uniformity")
                else:
                    report.append("  → STRONG BIAS: Significant deviation from uniformity")
            
            report.append("")
        
        # Cross-task analysis
        if 'cross_task_analysis' in results:
            report.append("🔄 CROSS-TASK COMPARISON")
            report.append("-" * 50)
            
            cross_results = results['cross_task_analysis']
            
            report.append("Position bias comparison:")
            for task_name, metrics in cross_results.items():
                report.append(f"\n{task_name}:")
                report.append(f"  • Beginning bias: {metrics['beginning_bias']*100:.1f}%")
                report.append(f"  • Middle bias: {metrics['middle_bias']*100:.1f}%")
                report.append(f"  • End bias: {metrics['end_bias']*100:.1f}%")
                report.append(f"  • Mean position: {metrics['mean_position']:.3f}")
        
        report.append("")
        
        # Key findings
        report.append("🔍 KEY FINDINGS")
        report.append("-" * 50)
        
        # Analyze cross-task patterns
        task_biases = {}
        if 'cross_task_analysis' in results:
            for task_name, metrics in results['cross_task_analysis'].items():
                beginning_pct = metrics['beginning_bias'] * 100
                end_pct = metrics['end_bias'] * 100
                middle_pct = metrics['middle_bias'] * 100
                
                if beginning_pct > 30:
                    task_biases[task_name] = "Beginning-biased"
                elif end_pct > 30:
                    task_biases[task_name] = "End-biased"
                elif middle_pct > 50:
                    task_biases[task_name] = "Middle-focused"
                else:
                    task_biases[task_name] = "Balanced"
        
        report.append("Position bias patterns by task:")
        for task_name, bias_type in task_biases.items():
            report.append(f"  • {task_name}: {bias_type}")
        
        report.append("")
        report.append("=" * 100)
        
        # Write report to file
        report_path = 'multi_task_position_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(report))
        
        print(f"📝 Comprehensive report saved to: {report_path}")
        
        # Also print to console
        print("\\n".join(report))

def main():
    """Main execution function"""
    
    # Define task files
    task_files = {
        "Passage Retrieval": {
            "file": "test_data/real_benchmarks/longbench_passage_retrieval_en.jsonl",
            "analyzer_class": PassageRetrievalAnalyzer
        },
        "PassKey Retrieval": {
            "file": "test_data/real_benchmarks/infinitebench_passkey.jsonl", 
            "analyzer_class": PassKeyAnalyzer
        },
        "KV Retrieval": {
            "file": "test_data/real_benchmarks/infinitebench_kv_retrieval.jsonl",
            "analyzer_class": KVRetrievalAnalyzer
        }
    }
    
    # Initialize multi-task analyzer
    multi_analyzer = MultiTaskPositionAnalyzer()
    
    # Add available tasks
    for task_name, task_config in task_files.items():
        file_path = task_config["file"]
        if Path(file_path).exists():
            analyzer = task_config["analyzer_class"](file_path)
            multi_analyzer.add_task(task_name, analyzer)
            print(f"✅ Added {task_name} analysis")
        else:
            print(f"⚠️ Warning: {file_path} not found, skipping {task_name}")
    
    if len(multi_analyzer.analyzers) == 0:
        print("❌ No valid task files found! Please check file paths.")
        return
    
    # Run comprehensive analysis
    results = multi_analyzer.run_analysis()
    
    print(f"\\n🎉 Analysis completed for {len(multi_analyzer.analyzers)} tasks!")
    print("📊 Check the generated visualization and report files.")

if __name__ == "__main__":
    main()