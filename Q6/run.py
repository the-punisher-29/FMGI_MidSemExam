import random
import string
import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import csv
import json

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ============================================================================
# PART 1: ORIGINAL DATASET
# ============================================================================

geography_qa = {
    1: {"question": "What is the deepest point in the Mariana Trench called?", "answer": "Challenger Deep"},
    2: {"question": "Which African country has three capital cities?", "answer": "South Africa"},
    3: {"question": "What is the name of the desert between Mongolia and China?", "answer": "Gobi Desert"},
    4: {"question": "Which strait separates Spain from Morocco?", "answer": "Strait of Gibraltar"},
    5: {"question": "What is the smallest country entirely within another country?", "answer": "Vatican City"},
    6: {"question": "Which river forms the border between Zambia and Zimbabwe?", "answer": "Zambezi River"},
    7: {"question": "What is the northernmost capital city in the world?", "answer": "Reykjavik"},
    8: {"question": "Which sea lies between Jordan and Israel?", "answer": "Dead Sea"},
    9: {"question": "What is the largest lake on the African continent?", "answer": "Lake Victoria"},
    10: {"question": "Which mountain range contains K2?", "answer": "Karakoram"},
    11: {"question": "What is the driest non-polar desert in the world?", "answer": "Atacama Desert"},
    12: {"question": "Which island nation is located in the Indian Ocean near Madagascar?", "answer": "Mauritius"},
    13: {"question": "What is the longest river in Europe?", "answer": "Volga River"},
    14: {"question": "Which country has the most time zones?", "answer": "France"},
    15: {"question": "What is the name of the sea between Australia and New Zealand?", "answer": "Tasman Sea"},
    16: {"question": "Which African country was formerly known as Abyssinia?", "answer": "Ethiopia"},
    17: {"question": "What is the highest waterfall in the world?", "answer": "Angel Falls"},
    18: {"question": "Which strait connects the Mediterranean Sea to the Atlantic Ocean?", "answer": "Strait of Gibraltar"},
    19: {"question": "What is the largest island in the Mediterranean Sea?", "answer": "Sicily"},
    20: {"question": "Which country contains the Okavango Delta?", "answer": "Botswana"},
    21: {"question": "What is the deepest lake in the world?", "answer": "Lake Baikal"},
    22: {"question": "Which mountain range separates Europe from Asia?", "answer": "Ural Mountains"},
    23: {"question": "What is the smallest ocean in the world?", "answer": "Arctic Ocean"},
    24: {"question": "Which country has the most UNESCO World Heritage Sites?", "answer": "Italy"},
    25: {"question": "What is the longest mountain range in the world?", "answer": "Andes"},
    26: {"question": "Which sea is located between Greece and Turkey?", "answer": "Aegean Sea"},
    27: {"question": "What is the largest salt flat in the world?", "answer": "Salar de Uyuni"},
    28: {"question": "Which river flows through Baghdad?", "answer": "Tigris River"},
    29: {"question": "What is the highest capital city in the world?", "answer": "La Paz"},
    30: {"question": "Which archipelago includes Java and Sumatra?", "answer": "Indonesian Archipelago"},
    31: {"question": "What is the westernmost point of continental Europe?", "answer": "Cabo da Roca"},
    32: {"question": "Which lake is shared by Uganda, Kenya, and Tanzania?", "answer": "Lake Victoria"},
    33: {"question": "What is the longest cave system in the world?", "answer": "Mammoth Cave"},
    34: {"question": "Which country owns the Faroe Islands?", "answer": "Denmark"},
    35: {"question": "What is the largest atoll in the world?", "answer": "Kiritimati"},
    36: {"question": "Which river delta is home to the Sundarbans mangrove forest?", "answer": "Ganges Delta"},
    37: {"question": "What is the southernmost city in the world?", "answer": "Ushuaia"},
    38: {"question": "Which sea separates Saudi Arabia from Africa?", "answer": "Red Sea"},
    39: {"question": "What is the largest landlocked country in the world?", "answer": "Kazakhstan"},
    40: {"question": "Which volcano destroyed Pompeii?", "answer": "Mount Vesuvius"},
    41: {"question": "What is the longest river in Australia?", "answer": "Murray River"},
    42: {"question": "Which country has the most islands?", "answer": "Sweden"},
    43: {"question": "What is the highest mountain in Africa?", "answer": "Mount Kilimanjaro"},
    44: {"question": "Which strait separates Sicily from mainland Italy?", "answer": "Strait of Messina"},
    45: {"question": "What is the largest bay in the world?", "answer": "Bay of Bengal"},
    46: {"question": "Which desert covers much of Botswana?", "answer": "Kalahari Desert"},
    47: {"question": "What is the deepest canyon in the world?", "answer": "Yarlung Tsangpo Grand Canyon"},
    48: {"question": "Which island group forms the westernmost point of Europe?", "answer": "Azores"},
    49: {"question": "What is the longest river in Asia?", "answer": "Yangtze River"},
    50: {"question": "Which country contains Angel Falls?", "answer": "Venezuela"}
}

# ============================================================================
# PART 2: NOISE GENERATION FUNCTIONS
# ============================================================================

def add_typos(text: str, level: str = "light") -> str:
    """Add keyboard-adjacent typos to text"""
    keyboard_adjacents = {
        'a': ['s', 'q', 'w', 'z'], 'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'], 'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 'r', 'd', 's'], 'f': ['d', 'r', 't', 'g', 'c', 'v'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'], 'h': ['g', 'y', 'u', 'j', 'b', 'n'],
        'i': ['u', 'o', 'k', 'j'], 'j': ['h', 'u', 'i', 'k', 'n', 'm'],
        'k': ['j', 'i', 'o', 'l', 'm'], 'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'p', 'l', 'k'], 'p': ['o', 'l'],
        'q': ['w', 'a'], 'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'], 't': ['r', 'y', 'g', 'f'],
        'u': ['y', 'i', 'j', 'h'], 'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'e', 's', 'a'], 'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'u', 'h', 'g'], 'z': ['a', 's', 'x']
    }
    
    num_typos = 2 if level == "light" else 4
    text_list = list(text)
    positions = []
    
    for i, char in enumerate(text_list):
        if char.lower() in keyboard_adjacents:
            positions.append(i)
    
    if positions:
        typo_positions = random.sample(positions, min(num_typos, len(positions)))
        for pos in typo_positions:
            char = text_list[pos].lower()
            if char in keyboard_adjacents:
                replacement = random.choice(keyboard_adjacents[char])
                if text_list[pos].isupper():
                    replacement = replacement.upper()
                text_list[pos] = replacement
    
    return ''.join(text_list)

def add_spacing_issues(text: str, level: str = "light") -> str:
    """Add spacing and punctuation issues"""
    if level == "light":
        words = text.split()
        for _ in range(2):
            if words:
                idx = random.randint(0, len(words) - 1)
                words[idx] = words[idx] + " "
        text = " ".join(words)
        if random.random() > 0.5 and text.endswith("?"):
            text = text[:-1]
    else:
        words = text.split()
        result = []
        for word in words:
            result.append(word)
            result.append(" " * random.randint(1, 3))
        text = "".join(result).strip()
        text = text.replace("?", "")
    
    return text

def add_unicode_confusables(text: str, level: str = "light") -> str:
    """Replace characters with Unicode look-alikes"""
    confusables = {
        'a': 'α', 'A': 'Α', 'o': 'о', 'O': 'О',
        'e': 'е', 'E': 'Е', 'i': 'і', 'I': 'І',
        'c': 'с', 'C': 'С', 'p': 'р', 'P': 'Р',
        'x': 'х', 'X': 'Х', 'y': 'у', 'Y': 'Υ',
        'M': 'Μ', 'N': 'Ν', 'B': 'Β', 'H': 'Η'
    }
    
    num_replacements = 2 if level == "light" else 4
    text_list = list(text)
    positions = []
    
    for i, char in enumerate(text_list):
        if char in confusables:
            positions.append(i)
    
    if positions:
        replace_positions = random.sample(positions, min(num_replacements, len(positions)))
        for pos in replace_positions:
            char = text_list[pos]
            if char in confusables:
                text_list[pos] = confusables[char]
    
    return ''.join(text_list)

def add_emoji(text: str, level: str = "light") -> str:
    """Add neutral emoji to text"""
    neutral_emoji = ['🌍', '📍', '🗺️', '🌊', '⛰️', '🏔️', '🌋', '🏝️', '🤔', '❓']
    
    num_emoji = 2 if level == "light" else 4
    words = text.split()
    
    if len(words) > 1:
        positions = random.sample(range(1, len(words)), min(num_emoji, len(words) - 1))
        for pos in sorted(positions, reverse=True):
            words.insert(pos, random.choice(neutral_emoji))
    
    return ' '.join(words)

# ============================================================================
# PART 3: GENERATE PERTURBED VARIANTS
# ============================================================================

def generate_all_perturbations(qa_dict: Dict) -> Dict:
    """Generate all perturbed variants for the dataset"""
    perturbed_data = {}
    
    noise_functions = {
        'typos': add_typos,
        'spacing': add_spacing_issues,
        'unicode': add_unicode_confusables,
        'emoji': add_emoji
    }
    
    for q_id, qa in qa_dict.items():
        question = qa['question']
        answer = qa['answer']
        
        # Store clean version
        perturbed_data[f"{q_id}_clean"] = {
            'id': f"{q_id}_clean",
            'original_id': q_id,
            'noise_type': 'clean',
            'noise_level': 'none',
            'question': question,
            'gold_answer': answer
        }
        
        # Generate perturbed versions
        for noise_type, noise_func in noise_functions.items():
            for level in ['light', 'heavy']:
                perturbed_q = noise_func(question, level)
                perturbed_data[f"{q_id}_{noise_type}_{level}"] = {
                    'id': f"{q_id}_{noise_type}_{level}",
                    'original_id': q_id,
                    'noise_type': noise_type,
                    'noise_level': level,
                    'question': perturbed_q,
                    'gold_answer': answer
                }
    
    return perturbed_data

# ============================================================================
# PART 4: EXPORT QUESTIONS FOR HF PLAYGROUND
# ============================================================================

def export_questions_for_hf(perturbed_data: Dict, filename: str = "questions_for_hf.json"):
    """Export all questions in a format ready for HF Playground"""
    questions_list = []
    
    for key, item in perturbed_data.items():
        questions_list.append({
            'id': item['id'],
            'question': item['question'],
            'gold_answer': item['gold_answer'],
            'noise_type': item['noise_type'],
            'noise_level': item['noise_level']
        })
    
    # Save to JSON for easy copying
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(questions_list, f, ensure_ascii=False, indent=2)
    
    # Also create a simple text file with just the questions
    with open("questions_only.txt", 'w', encoding='utf-8') as f:
        for q in questions_list:
            f.write(f"ID: {q['id']}\n")
            f.write(f"Question: {q['question']}\n")
            f.write(f"---\n")
    
    print(f"Exported {len(questions_list)} questions to {filename}")
    return questions_list

# ============================================================================
# PART 5: PLACEHOLDER FOR MODEL RESPONSES
# ============================================================================

def load_hf_responses(filename: str = "hf_responses.json") -> Dict:
    with open(filename, "r", encoding="utf-8") as f:
        hf_responses = json.load(f)
    return hf_responses

# ============================================================================
# PART 6: EVALUATION WITH HF RESPONSES
# ============================================================================

def evaluate_hf_responses(perturbed_data: Dict, hf_responses: Dict) -> pd.DataFrame:
    """Evaluate HF model responses and create results dataframe"""
    results = []
    
    for key, item in perturbed_data.items():
        # Get model prediction from HF responses
        if key in hf_responses:
            pred = hf_responses[key].get('response', 'NO_RESPONSE')
        else:
            pred = 'MISSING_RESPONSE'  # Flag missing responses
        
        # Check if correct (case-insensitive, trimmed comparison)
        gold_normalized = item['gold_answer'].lower().strip()
        pred_normalized = pred.lower().strip() if pred else ""
        correct = 1 if gold_normalized == pred_normalized else 0
        
        results.append({
            'id': item['id'],
            'original_id': item['original_id'],
            'noise_type': item['noise_type'],
            'noise_level': item['noise_level'],
            'prompt_in': item['question'],
            'gold': item['gold_answer'],
            'pred': pred,
            'correct': correct
        })
    
    return pd.DataFrame(results)

# ============================================================================
# PART 7: ROBUSTNESS INTERVENTION TEMPLATES
# ============================================================================

def preprocess_input(text: str) -> str:
    """Light preprocessor to clean input text"""
    # Normalize Unicode to ASCII equivalents
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove emoji and special characters
    text = re.sub(r'[^\w\s\?\.\,\-]', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure question mark at end
    text = text.strip()
    if not text.endswith('?'):
        text = text + '?'
    
    return text

def robust_prompt_template(question: str) -> str:
    """Apply robust prompting template"""
    template = """INSTRUCTION: You are answering geography questions that may contain typos, unusual spacing, Unicode characters, or emoji.

IMPORTANT:
- Mentally normalize any visual variations (α→a, о→o, emojis→ignore)
- Focus on the semantic intent of the question
- Ignore extra spaces and punctuation errors
- If you recognize a misspelled geographic term, use the correct spelling in your answer

Answer format: Provide ONLY the specific answer requested (location, country, or geographic feature name). No explanations.

Question: {question}
Answer:"""
    
    return template.format(question=question)

def export_robust_questions(perturbed_data: Dict, subset_size: int = 20):
    """Export questions with robust prompting for HF testing"""
    # Sample subset
    subset_keys = random.sample(list(perturbed_data.keys()), min(subset_size * 9, len(perturbed_data)))
    
    robust_questions = []
    for key in subset_keys:
        item = perturbed_data[key]
        
        # Apply preprocessing
        cleaned_question = preprocess_input(item['question'])
        
        # Apply robust prompt template
        robust_prompt = robust_prompt_template(cleaned_question)
        
        robust_questions.append({
            'id': item['id'] + '_robust',
            'original_id': item['original_id'],
            'noise_type': item['noise_type'],
            'noise_level': item['noise_level'],
            'original_question': item['question'],
            'cleaned_question': cleaned_question,
            'robust_prompt': robust_prompt,
            'gold_answer': item['gold_answer']
        })
    
    # Save robust questions
    with open('robust_questions_for_hf.json', 'w', encoding='utf-8') as f:
        json.dump(robust_questions, f, ensure_ascii=False, indent=2)
    
    print(f"Exported {len(robust_questions)} robust questions")
    return robust_questions

# ============================================================================
# PART 8: VISUALIZATION AND REPORTING
# ============================================================================

def create_accuracy_heatmap(df: pd.DataFrame, title: str = "Accuracy by Noise Type and Level"):
    """Create a heatmap showing accuracy by noise type and level"""
    accuracy_matrix = df.groupby(['noise_type', 'noise_level'])['correct'].mean() * 100
    
    pivot_table = accuracy_matrix.reset_index().pivot(
        index='noise_type', 
        columns='noise_level', 
        values='correct'
    )
    
    column_order = ['none', 'light', 'heavy']
    row_order = ['clean', 'typos', 'spacing', 'unicode', 'emoji']
    
    pivot_table = pivot_table.reindex(
        columns=[col for col in column_order if col in pivot_table.columns],
        index=[idx for idx in row_order if idx in pivot_table.index]
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'})
    plt.title(title)
    plt.xlabel('Noise Level')
    plt.ylabel('Noise Type')
    plt.tight_layout()
    return plt

def generate_error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze errors by category"""
    errors = df[df['correct'] == 0].copy()
    
    error_summary = []
    for noise_type in errors['noise_type'].unique():
        for noise_level in errors['noise_level'].unique():
            mask = (errors['noise_type'] == noise_type) & (errors['noise_level'] == noise_level)
            subset = errors[mask]
            
            if len(subset) > 0:
                error_summary.append({
                    'Noise Type': noise_type,
                    'Noise Level': noise_level,
                    'Error Count': len(subset),
                    'Error Rate (%)': (len(subset) / len(df[mask])) * 100 if len(df[mask]) > 0 else 0,
                    'Sample Errors': subset[['prompt_in', 'gold', 'pred']].head(2).to_dict('records')
                })
    
    return pd.DataFrame(error_summary)

def create_degradation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary table of accuracy degradation"""
    baseline = df[df['noise_type'] == 'clean']['correct'].mean() * 100
    
    summary_data = []
    for noise_type in ['typos', 'spacing', 'unicode', 'emoji']:
        for level in ['light', 'heavy']:
            mask = (df['noise_type'] == noise_type) & (df['noise_level'] == level)
            if mask.any():
                accuracy = df[mask]['correct'].mean() * 100
                degradation = accuracy - baseline
                
                summary_data.append({
                    'Noise Type': noise_type.capitalize(),
                    'Level': level.capitalize(),
                    'Accuracy (%)': f"{accuracy:.1f}",
                    'Degradation (pp)': f"{degradation:.1f}"
                })
    
    return pd.DataFrame(summary_data)


# ============================================================================
# MAIN EXECUTION WORKFLOW
# ============================================================================

def main():
    print("="*60)
    print("ROBUSTNESS TO MESSY INPUTS STUDY - HF PLAYGROUND VERSION")
    print("="*60)
    
    # STEP 1: Generate perturbed data
    print("\nSTEP 1: Generating perturbed variants...")
    perturbed_data = generate_all_perturbations(geography_qa)
    print(f"✓ Generated {len(perturbed_data)} total variants")
    
    # STEP 2: Export questions for HF Playground
    print("\nSTEP 2: Exporting questions for HF Playground...")
    questions_list = export_questions_for_hf(perturbed_data)
    print("✓ Questions exported to 'questions_for_hf.json' and 'questions_only.txt'")
    
    print("\n" + "="*60)
    print("ACTION REQUIRED:")
    print("1. Copy questions from 'questions_for_hf.json'")
    print("2. Run them through HF Playground")
    print("3. Save responses in the format shown in load_hf_responses()")
    print("="*60)
    
    # STEP 3: Load HF responses (YOU NEED TO FILL THIS)
    print("\nSTEP 3: Loading HF responses...")
    hf_responses = load_hf_responses()  # <-- FILL THIS WITH YOUR RESPONSES
    
    if not hf_responses:
        print("\n⚠️  WARNING: No HF responses loaded!")
        print("Please add your HF responses to the load_hf_responses() function")
        print("Continuing with placeholder data for demonstration...\n")
        
        # Create dummy responses for demonstration
        hf_responses = {}
        for key in perturbed_data.keys():
            # Simulate responses (replace with actual)
            if 'clean' in key:
                hf_responses[key] = {'response': perturbed_data[key]['gold_answer']}
            else:
                # Simulate some errors
                if random.random() > 0.7:
                    hf_responses[key] = {'response': 'unknown'}
                else:
                    hf_responses[key] = {'response': perturbed_data[key]['gold_answer']}
    
    # STEP 4: Evaluate responses
    print("\nSTEP 4: Evaluating HF model responses...")
    df_results = evaluate_hf_responses(perturbed_data, hf_responses)
    print(f"✓ Evaluated {len(df_results)} prompts")
    
    # STEP 5: Calculate metrics
    print("\nSTEP 5: Performance Metrics:")
    baseline_acc = df_results[df_results['noise_type'] == 'clean']['correct'].mean() * 100
    print(f"Clean accuracy: {baseline_acc:.1f}%")
    
    for noise_type in ['typos', 'spacing', 'unicode', 'emoji']:
        print(f"\n{noise_type.capitalize()}:")
        for level in ['light', 'heavy']:
            mask = (df_results['noise_type'] == noise_type) & (df_results['noise_level'] == level)
            if mask.any():
                acc = df_results[mask]['correct'].mean() * 100
                print(f"  {level}: {acc:.1f}% (degradation: {acc - baseline_acc:+.1f}pp)")
    
    # STEP 6: Export robust questions for testing
    print("\nSTEP 6: Generating robust prompts for intervention testing...")
    robust_questions = export_robust_questions(perturbed_data, subset_size=20)
    print("✓ Robust questions exported to 'robust_questions_for_hf.json'")
    
    print("\n" + "="*60)
    print("ACTION REQUIRED FOR ROBUST TESTING:")
    print("1. Copy prompts from 'robust_questions_for_hf.json'")
    print("2. Test the 'robust_prompt' field in HF Playground")
    print("3. Compare results with baseline")
    print("="*60)
    
    # STEP 7: Generate visualizations
    print("\nSTEP 7: Generating visualizations...")
    
    # Only create visualizations if we have real data
    if len(df_results) > 0 and df_results['correct'].notna().any():
        # Create accuracy heatmap
        plt_heatmap = create_accuracy_heatmap(df_results)
        plt_heatmap.savefig('hf_accuracy_heatmap.png', dpi=150, bbox_inches='tight')
        plt_heatmap.show()
        print("✓ Saved heatmap to 'hf_accuracy_heatmap.png'")
        
        # Create summary table
        summary_df = create_degradation_summary(df_results)
        print("\nPerformance Summary:")
        print(summary_df.to_string(index=False))
        
        # Error analysis
        error_df = generate_error_analysis(df_results)
        print("\nError Analysis Summary:")
        print(f"Total errors: {(df_results['correct'] == 0).sum()}")
        print(f"Overall error rate: {((df_results['correct'] == 0).sum() / len(df_results)) * 100:.1f}%")
    
    # STEP 8: Save results
    print("\nSTEP 8: Saving results...")
    df_results.to_csv('hf_robustness_results.csv', index=False)
    print("✓ Results saved to 'hf_robustness_results.csv'")
    
    # STEP 9: Instructions for completing the study
    print("\n" + "="*60)
    print("NEXT STEPS TO COMPLETE THE STUDY:")
    print("="*60)
    print("1. Fill in actual HF responses in load_hf_responses()")
    print("2. Test robust prompts and add those responses")
    print("3. Re-run this script with actual data")
    print("4. Compare baseline vs robust intervention results")
    print("5. Document findings and error patterns")
    print("="*60)
    
    return df_results, perturbed_data

if __name__ == "__main__":
    results, perturbed_data = main()
    
    print("\n✅ Script completed! Check the exported files and follow the instructions above.")