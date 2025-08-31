import torch
from bert_score import score, BERTScorer
import numpy as np

def calculate_bert_scores(golden_labels, 
                         generated_texts,
                         model_type = "microsoft/deberta-xlarge-mnli",
                         lang = "en",
                         verbose = False,
                         device = 'cuda'):
   
    
    if len(golden_labels) != len(generated_texts):
        raise ValueError("Number of golden labels must match number of generated texts")
    
    
    print(f"Using device: {device}")
    print(f"Using model: {model_type}")
    
    # Calculate BERTScore
    P, R, F1 = score(
        cands=generated_texts,
        refs=golden_labels,
        model_type=model_type,
        lang=lang,
        verbose=verbose,
        device=device
    )
    
    # Convert tensors to lists
    precision_scores = P.tolist()
    recall_scores = R.tolist()
    f1_scores = F1.tolist()
    
    return {
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'f1_scores': f1_scores,
        'avg_precision': np.mean(precision_scores),
        'avg_recall': np.mean(recall_scores),
        'avg_f1': np.mean(f1_scores),
        'max_f1': np.max(f1_scores),
        'min_f1': np.min(f1_scores)
    }


def print_bert_results(golden_labels, 
                      generated_texts, 
                      results):
    """Print detailed BERTScore results"""
    
    print("="*80)
    print("BERTSCORE EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"Average Recall:    {results['avg_recall']:.4f}")
    print(f"Average F1:        {results['avg_f1']:.4f}")
    print(f"Max F1 Score:      {results['max_f1']:.4f}")
    print(f"Min F1 Score:      {results['min_f1']:.4f}")
    
    print(f"\nIndividual Scores:")
    print("-" * 80)
    print(f"{'Pair':<4} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Quality':<12}")
    print("-" * 80)
    
    for i, (golden, generated, p, r, f1) in enumerate(zip(
        golden_labels, generated_texts, 
        results['precision_scores'], results['recall_scores'], results['f1_scores']
    )):
        # Determine quality level based on F1 score
        if f1 >= 0.9:
            quality = "Excellent"
        elif f1 >= 0.8:
            quality = "Very Good"
        elif f1 >= 0.7:
            quality = "Good"
        elif f1 >= 0.6:
            quality = "Fair"
        elif f1 >= 0.5:
            quality = "Poor"
        else:
            quality = "Very Poor"
        
        print(f"{i+1:<4} {p:<10.4f} {r:<10.4f} {f1:<10.4f} {quality}")
        print(f"Golden:    {golden[:70]}{'...' if len(golden) > 70 else ''}")
        print(f"Generated: {generated[:70]}{'...' if len(generated) > 70 else ''}")
        print()


# Example usage and test function
def main():
    golden_labels = #TODO
    
    generated_texts = #TODO
    
    print("BERTScore Evaluation:")
    
    # Calculate BERTScore with default model
    results = calculate_bert_scores(golden_labels, generated_texts, verbose=True)
    print_bert_results(golden_labels, generated_texts, results)
    


if __name__ == "__main__":
    main()