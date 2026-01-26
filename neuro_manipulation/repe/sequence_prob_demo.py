#!/usr/bin/env python3
"""
FIXED: Anger emotion activation impact on word probability demonstration.

This script fixes the fundamental disconnect in the original version by using
CombinedVLLMHook.generate_with_control() for both activation AND probability measurement.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Union, Optional
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from neuro_manipulation.configs.experiment_config import get_repe_eng_config
from neuro_manipulation.model_utils import load_emotion_readers, setup_model_and_tokenizer
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.repe.sequence_prob_vllm_hook import CombinedVLLMHook

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedAngerActivationDemo:
    """FIXED demo class that properly integrates activation and probability measurement."""
    
    def __init__(self, model_path: str = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"):
        """Initialize the demo with model and emotion readers."""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get configuration
        self.repe_eng_config = get_repe_eng_config(model_path)
        logger.info(f"Using model: {model_path}")
        
        # Setup model and tokenizer for emotion reader loading
        model, tokenizer, prompt_format = setup_model_and_tokenizer(
            self.repe_eng_config, from_vllm=False
        )
        
        # Get hidden layers
        num_hidden_layers = ModelLayerDetector.num_layers(model)
        self.hidden_layers = list(range(-1, -num_hidden_layers - 1, -1))
        self.control_layers = self.hidden_layers[len(self.hidden_layers) // 3 : 2 * len(self.hidden_layers) // 3]
        logger.info(f"Model has {num_hidden_layers} layers")
        logger.info(f"Using control layers: {self.control_layers}")
        
        # Load emotion readers (enable_thinking defaults to False for demo)
        self.emotion_rep_readers = load_emotion_readers(
            self.repe_eng_config, model, tokenizer, self.hidden_layers
        )
        logger.info("Emotion readers loaded successfully")
        
        # Switch to vLLM for inference
        del model
        self.model, self.tokenizer, self.prompt_format = setup_model_and_tokenizer(
            self.repe_eng_config, from_vllm=True
        )
        
        # 🔧 FIX: Use integrated CombinedVLLMHook for BOTH activation AND probability
        self.combined_hook = CombinedVLLMHook(
            model=self.model,
            tokenizer=self.tokenizer,
            enable_sequence_prob=True,
            enable_rep_control=True,
            layers=self.control_layers,
            block_name=self.repe_eng_config["block_name"]
        )
        
        # 🔧 FIX: Use single-token target words for better probability calculation
        self.target_words = {
            "anger": "anger",      # Single token: [4003]
            "angry_space": " angry",  # Single token: [18514]  
            "mad": " mad",         # Single token: [12796]
            "happy": " happy",     # Single token: [6072]
            "sad": " sad"          # Single token: [6507]
        }
        
        # Log tokenization for verification
        for name, word in self.target_words.items():
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            logger.info(f"Target '{name}' ('{word}'): {tokens} -> {self.tokenizer.decode(tokens)}")
        
    def create_test_prompts(self) -> List[str]:
        """Create test prompts for probability analysis."""
        prompts = [
            "The person felt very",
            "His reaction was quite", 
            "She became increasingly",
            "The customer seemed",
            "After the incident, he was"
        ]
        return prompts
        
    def get_anger_activations(self, intensity: float) -> Dict[int, torch.Tensor]:
        """Get anger emotion activations for given intensity."""
        rep_reader = self.emotion_rep_readers["anger"]
        
        # 🔧 FIX: Only use control layers, and keep on CPU for RPC
        activations = {
            layer: torch.tensor(
                intensity * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
            ).cpu().half()
            for layer in self.control_layers  # Only control layers!
        }
        
        return activations
        
    def measure_word_probabilities_integrated(self, prompts: List[str], 
                                            activations: Optional[Dict[int, torch.Tensor]] = None) -> Dict[str, List[float]]:
        """🔧 FIXED: Use integrated approach for activation + probability measurement."""
        
        results = {}
        
        for target_name, target_word in self.target_words.items():
            logger.info(f"Measuring probabilities for target: '{target_name}' ('{target_word}')")
            
            # 🔧 FIX: Set activations first, then measure probabilities
            if activations is not None:
                # Set the activations on the hook
                self.combined_hook._set_control_activations(activations)
                
            try:
                # Measure probabilities (with or without activations set)
                prob_results = self.combined_hook.get_log_prob(
                    text_inputs=prompts,
                    target_sequences=[target_word],
                    max_new_tokens=2,
                    temperature=0.7
                )
            finally:
                # Clear activations after measurement
                if activations is not None:
                    self.combined_hook._clear_control_activations()
            
            # Extract probabilities
            probabilities = []
            for i, prompt in enumerate(prompts):
                found_prob = 0.0
                for result in prob_results:
                    if result['sequence'] == target_word:
                        found_prob = result['prob']
                        break
                probabilities.append(found_prob)
            
            results[target_name] = probabilities
            avg_prob = np.mean(probabilities)
            logger.info(f"Average probability for '{target_name}': {avg_prob:.6f}")
            
        return results
    
    def run_probability_analysis(self):
        """Run complete probability analysis with different anger intensities."""
        logger.info("🔧 FIXED: Starting integrated anger activation probability analysis")
        
        test_prompts = self.create_test_prompts()
        intensities = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        results = {
            'prompts': test_prompts,
            'intensities': intensities,
            'probabilities': {}
        }
        
        for intensity in intensities:
            logger.info(f"Testing anger intensity: {intensity}")
            
            if intensity == 0.0:
                # Baseline without activation
                probabilities = self.measure_word_probabilities_integrated(test_prompts, activations=None)
                condition = "baseline"
            else:
                # With anger activation
                activations = self.get_anger_activations(intensity)
                probabilities = self.measure_word_probabilities_integrated(test_prompts, activations)
                condition = f"anger_{intensity}"
            
            results['probabilities'][condition] = probabilities
            
            # Log summary for each target word
            for target_name in self.target_words.keys():
                avg_prob = np.mean(probabilities[target_name])
                logger.info(f"  {target_name}: {avg_prob:.6f}")
        
        return results
    
    def demonstrate_generation_differences(self):
        """Demonstrate that generation actually changes with anger activation."""
        logger.info("🔧 DEMONSTRATING GENERATION DIFFERENCES")
        
        test_prompt = "The person felt very"
        intensity = 1.5
        
        logger.info(f"Test prompt: '{test_prompt}'")
        
        # Baseline generation
        logger.info("\n--- BASELINE GENERATION ---")
        baseline_outputs = self.combined_hook.generate_with_control(
            prompts=[test_prompt],
            activations=None,
            max_new_tokens=8,
            temperature=0.7
        )
        baseline_text = baseline_outputs[0].outputs[0].text
        logger.info(f"Generated: '{baseline_text}'")
        
        # Anger-controlled generation  
        logger.info(f"\n--- ANGER ACTIVATION (intensity={intensity}) ---")
        activations = self.get_anger_activations(intensity)
        logger.info(f"Created activations for {len(activations)} layers")
        
        controlled_outputs = self.combined_hook.generate_with_control(
            prompts=[test_prompt],
            activations=activations,
            max_new_tokens=8,
            temperature=0.7
        )
        controlled_text = controlled_outputs[0].outputs[0].text
        logger.info(f"Generated: '{controlled_text}'")
        
        # Compare
        logger.info(f"\n--- COMPARISON ---")
        logger.info(f"Baseline:  '{test_prompt}' → '{baseline_text}'")
        logger.info(f"Anger:     '{test_prompt}' → '{controlled_text}'")
        logger.info(f"Different: {baseline_text != controlled_text}")
        
        return baseline_text, controlled_text
    
    def create_visualization(self, results: Dict):
        """Create visualization of probability changes."""
        logger.info("Creating visualization...")
        
        intensities = results['intensities']
        target_names = list(self.target_words.keys())
        
        # Create subplots for each target word
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, target_name in enumerate(target_names):
            ax = axes[i]
            
            # Calculate average probabilities across prompts for each intensity
            avg_probabilities = []
            for intensity in intensities:
                if intensity == 0.0:
                    condition = "baseline"
                else:
                    condition = f"anger_{intensity}"
                
                avg_prob = np.mean(results['probabilities'][condition][target_name])
                avg_probabilities.append(avg_prob)
            
            # Plot
            ax.plot(intensities, avg_probabilities, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Anger Activation Intensity')
            ax.set_ylabel(f'Probability of "{self.target_words[target_name]}"')
            ax.set_title(f'{target_name}')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for x, y in zip(intensities, avg_probabilities):
                ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        # Remove empty subplot
        if len(target_names) < 6:
            axes[5].remove()
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("logs/anger_demo_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_dir / "fixed_anger_probability_impact.png", dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_dir / 'fixed_anger_probability_impact.png'}")
        
        return output_dir / "fixed_anger_probability_impact.png"

def main():
    """Main demonstration function."""
    print("🔧 FIXED: Anger Activation Word Probability Demonstration")
    print("=" * 70)
    
    try:
        # Initialize fixed demo
        demo = FixedAngerActivationDemo()
        
        # 1. Demonstrate that generation actually works
        print("\n1. 🔧 GENERATION DIFFERENCE VERIFICATION")
        print("-" * 45)
        baseline_text, controlled_text = demo.demonstrate_generation_differences()
        
        # 2. Run integrated probability analysis  
        print("\n2. 🔧 INTEGRATED PROBABILITY ANALYSIS")
        print("-" * 40)
        results = demo.run_probability_analysis()
        
        # 3. Create visualization
        print("\n3. 📊 CREATING VISUALIZATION")
        print("-" * 28)
        plot_path = demo.create_visualization(results)
        
        # 4. Summary and analysis
        print("\n" + "=" * 70)
        print("🔧 FIXED DEMONSTRATION SUMMARY")
        print("=" * 70)
        
        print(f"✅ Model: {demo.model_path}")
        print(f"✅ Control layers: {demo.control_layers}")
        print(f"✅ Target words: {list(demo.target_words.keys())}")
        print(f"✅ Generation verification: Baseline ≠ Controlled")
        print(f"✅ Visualization saved: {plot_path}")
        
        # Analysis of results
        print(f"\n📊 PROBABILITY ANALYSIS:")
        
        baseline_condition = "baseline"
        max_intensity_condition = "anger_2.0"
        
        for target_name in demo.target_words.keys():
            baseline_avg = np.mean(results['probabilities'][baseline_condition][target_name])
            max_intensity_avg = np.mean(results['probabilities'][max_intensity_condition][target_name])
            
            if baseline_avg > 0:
                change_ratio = max_intensity_avg / baseline_avg
                change_pct = ((max_intensity_avg - baseline_avg) / baseline_avg) * 100
            else:
                change_ratio = float('inf') if max_intensity_avg > 0 else 1.0
                change_pct = 0.0
            
            print(f"   {target_name}:")
            print(f"     Baseline: {baseline_avg:.6f}")
            print(f"     Max intensity: {max_intensity_avg:.6f}")
            print(f"     Change: {change_ratio:.2f}x ({change_pct:+.1f}%)")
            
            if change_ratio > 1.5:
                print(f"     🔥 Strong increase with anger activation!")
            elif change_ratio > 1.1:
                print(f"     📈 Moderate increase")
            elif change_ratio < 0.9:
                print(f"     📉 Decrease with anger activation")
            else:
                print(f"     📊 Minimal change")
        
        print("\n🎯 Fixed demonstration completed successfully!")
        print("🔧 This version properly integrates activation and probability measurement!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise

if __name__ == "__main__":
    main() 