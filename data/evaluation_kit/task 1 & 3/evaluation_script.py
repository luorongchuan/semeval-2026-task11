import json
import math
from typing import List, Dict, Any, Tuple, Optional, Callable

# --- 1. MOCK DATA FOR DEMONSTRATION ---

# Ground Truth Data
GROUND_TRUTH_DATA_JSON = """
[
    {"id": "id1", "validity": false, "plausibility": true},   
    {"id": "id2", "validity": true, "plausibility": true},    
    {"id": "id3", "validity": false, "plausibility": false},  
    {"id": "id4", "validity": true, "plausibility": false},   
    {"id": "id5", "validity": true, "plausibility": true},    
    {"id": "id6", "validity": false, "plausibility": true},   
    {"id": "id7", "validity": false, "plausibility": false},  
    {"id": "id8", "validity": true, "plausibility": false},   
    {"id": "id9", "validity": true, "plausibility": true},    
    {"id": "id10", "validity": false, "plausibility": true}    
]
"""
# Mock Predictions for Model 0 (Perfect Score)
MOCK_PREDICTIONS_MODEL_0_JSON = """
[
    {"id": "id1", "validity": false},  
    {"id": "id2", "validity": true},   
    {"id": "id3", "validity": false}, 
    {"id": "id4", "validity": true},  
    {"id": "id5", "validity": true},  
    {"id": "id6", "validity": false},   
    {"id": "id7", "validity": false},  
    {"id": "id8", "validity": true},  
    {"id": "id9", "validity": true},  
    {"id": "id10", "validity": false}   
]
"""
# Mock Predictions for Model 1 (High Bias, 70% Acc)
MOCK_PREDICTIONS_MODEL_1_JSON = """
[
    {"id": "id1", "validity": false},  
    {"id": "id2", "validity": true},   
    {"id": "id3", "validity": false}, 
    {"id": "id4", "validity": false},  
    {"id": "id5", "validity": true},  
    {"id": "id6", "validity": true},   
    {"id": "id7", "validity": false},  
    {"id": "id8", "validity": true},  
    {"id": "id9", "validity": true},  
    {"id": "id10", "validity": true}   
]
"""

# Mock Predictions for Model 2 (Lower Bias, 70% Acc)
MOCK_PREDICTIONS_MODEL_2_JSON = """
[
    {"id": "id1", "validity": false},  
    {"id": "id2", "validity": true},   
    {"id": "id3", "validity": false}, 
    {"id": "id4", "validity": true},  
    {"id": "id5", "validity": false},  
    {"id": "id6", "validity": false},   
    {"id": "id7", "validity": true},  
    {"id": "id8", "validity": false},  
    {"id": "id9", "validity": true},  
    {"id": "id10", "validity": false}   
]
"""

# --- 2. CORE FUNCTIONS FOR ACCURACY AND BIAS CALCULATION ---

def calculate_accuracy(
    ground_truth_list: List[Dict[str, Any]],
    predictions_list: List[Dict[str, Any]],
    metric_name: str,
    prediction_key: str,
    plausibility_filter: Optional[bool] = None # Filter by ground truth plausibility
) -> Tuple[float, int, int]:
    """
    Calculates the accuracy of 'validity' predictions against ground truth labels,
    with an optional filter based on ground truth 'plausibility'.
    """
    # Map ground truth for easy lookup by ID
    gt_map = {item['id']: item for item in ground_truth_list}

    correct_predictions = 0
    total_predictions = 0

    for pred_item in predictions_list:
        item_id = pred_item['id']
        
        if item_id in gt_map:
            gt_item = gt_map[item_id]
            
            # Apply plausibility filter (grouping)
            gt_plausibility = gt_item.get('plausibility')
            # Check if we should skip the item based on the plausibility filter
            if plausibility_filter is not None and gt_plausibility != plausibility_filter:
                continue 
            
            # Ensure the required keys exist and are booleans
            if metric_name in gt_item and prediction_key in pred_item:
                true_label = gt_item[metric_name]
                predicted_label = pred_item[prediction_key]

                if isinstance(true_label, bool) and isinstance(predicted_label, bool):
                    total_predictions += 1
                    if true_label == predicted_label:
                        correct_predictions += 1

    if total_predictions == 0:
        return 0.0, 0, 0

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, correct_predictions, total_predictions


def calculate_subgroup_accuracy(
    gt_map: Dict[str, Any],
    predictions_list: List[Dict[str, Any]],
    gt_validity: bool,
    gt_plausibility: bool
) -> Tuple[float, int, int]:
    """
    Calculates accuracy for a specific subgroup defined by ground truth validity AND plausibility.
    """
    correct_predictions = 0
    total_predictions = 0
    
    for pred_item in predictions_list:
        item_id = pred_item['id']
        if item_id in gt_map:
            gt_item = gt_map[item_id]
            
            # Filter by both ground truth labels
            if gt_item.get('validity') == gt_validity and gt_item.get('plausibility') == gt_plausibility:
                if 'validity' in gt_item and 'validity' in pred_item:
                    true_label = gt_item['validity']
                    predicted_label = pred_item['validity']
                    
                    if isinstance(true_label, bool) and isinstance(predicted_label, bool):
                        total_predictions += 1
                        if true_label == predicted_label:
                            correct_predictions += 1

    if total_predictions == 0:
        return 0.0, 0, 0

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, correct_predictions, total_predictions


def calculate_content_effect_bias(accuracies: Dict[str, float]) -> Dict[str, float]:
    """
    Calculates the content effect bias metrics as defined.
    """
    
    acc_plausible_valid = accuracies.get('acc_plausible_valid', 0.0)
    acc_implausible_valid = accuracies.get('acc_implausible_valid', 0.0)
    acc_plausible_invalid = accuracies.get('acc_plausible_invalid', 0.0)
    acc_implausible_invalid = accuracies.get('acc_implausible_invalid', 0.0)

    # 1. content_effect_intra_validity_label
    intra_valid_diff = abs(acc_plausible_valid - acc_implausible_valid)
    intra_invalid_diff = abs(acc_plausible_invalid - acc_implausible_invalid)
    content_effect_intra_validity_label = (intra_valid_diff + intra_invalid_diff) / 2.0

    # 2. content_effect_inter_validity_label
    inter_plausible_diff = abs(acc_plausible_valid - acc_plausible_invalid)
    inter_implausible_diff = abs(acc_implausible_valid - acc_implausible_invalid)
    content_effect_inter_validity_label = (inter_plausible_diff + inter_implausible_diff) / 2.0

    # 3. tot_content_effect
    tot_content_effect = (content_effect_intra_validity_label + content_effect_inter_validity_label) / 2.0
    
    return {
        'content_effect_intra_validity_label': content_effect_intra_validity_label,
        'content_effect_inter_validity_label': content_effect_inter_validity_label,
        'tot_content_effect': tot_content_effect
    }

def calculate_smooth_combined_metric(overall_accuracy: float, total_content_effect: float) -> float:
    """
    Computes a smooth combined score using the natural logarithm.
    Formula: accuracy / (1 + ln(1 + content_effect))
    """
    if total_content_effect < 0:
        return 0.0

    log_penalty = math.log(1 + total_content_effect)
    combined_smooth_score = overall_accuracy / (1 + log_penalty)
    return combined_smooth_score

def analyze_model_performance(model_name: str, ground_truth: List[Dict[str, Any]], predictions: List[Dict[str, Any]], gt_map: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the full analysis pipeline for a single model's prediction set and returns key metrics."""
    
    print("\n" + "#" * 70)
    print(f"--- RESULTS FOR {model_name} ---")
    print("#" * 70)

    # Define the common arguments for validity calculation
    common_args = {
        'ground_truth_list': ground_truth,
        'predictions_list': predictions,
        'metric_name': 'validity',
        'prediction_key': 'validity'
    }

    # --- 1. Overall Accuracy ---
    overall_acc, overall_correct, overall_total = calculate_accuracy(
        **common_args,
        plausibility_filter=None
    )
    print(f"Overall Data Count: {overall_total}")
    print("-" * 50)
    print(f"[Group: ALL] Validity Accuracy: {overall_acc:.2f}% ({overall_correct} / {overall_total})")

    # --- 2. Plausible/Implausible Group Accuracy ---
    plausible_acc, plausible_correct, plausible_total = calculate_accuracy(
        **common_args,
        plausibility_filter=True
    )
    print(f"\n[Group: PLAUSIBLE (Plausibility: True)] Validity Accuracy: {plausible_acc:.2f}% ({plausible_correct} / {plausible_total})")

    implausible_acc, implausible_correct, implausible_total = calculate_accuracy(
        **common_args,
        plausibility_filter=False
    )
    print(f"[Group: IMPLAUSIBLE (Plausibility: False)] Validity Accuracy: {implausible_acc:.2f}% ({implausible_correct} / {implausible_total})")
    
    # --- 3. Content Effect Bias Calculation ---
    print("\n" + "=" * 50)
    print("--- Content Effect Bias Metrics ---")
    
    # Calculate the four required conditional accuracies
    acc_plausible_valid, _, total_pv = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=True, gt_plausibility=True)
    acc_implausible_valid, _, total_iv = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=True, gt_plausibility=False)
    acc_plausible_invalid, _, total_pi = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=False, gt_plausibility=True)
    acc_implausible_invalid, _, total_ii = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=False, gt_plausibility=False)
    
    conditional_accuracies = {
        'acc_plausible_valid': acc_plausible_valid,
        'acc_implausible_valid': acc_implausible_valid,
        'acc_plausible_invalid': acc_plausible_invalid,
        'acc_implausible_invalid': acc_implausible_invalid
    }

    # Print conditional results for clarity
    print(f"\nConditional Accuracies (Validity Prediction):")
    print(f"  Valid & Plausible (V/P) (N={total_pv}): {acc_plausible_valid:.2f}%")
    print(f"  Valid & Implausible (V/I) (N={total_iv}): {acc_implausible_valid:.2f}%")
    print(f"  Invalid & Plausible (IV/P) (N={total_pi}): {acc_plausible_invalid:.2f}%")
    print(f"  Invalid & Implausible (IV/I) (N={total_ii}): {acc_implausible_invalid:.2f}%")

    # Calculate and print bias metrics
    bias_metrics = calculate_content_effect_bias(conditional_accuracies)
    tot_content_effect = bias_metrics['tot_content_effect']

    print("\nCalculated Content Effect Metrics (Bias towards Plausibility):")
    print(f"  Intra-Validity Bias (acc_intra): {bias_metrics['content_effect_intra_validity_label']:.2f}%")
    print(f"  Inter-Validity Bias (acc_inter): {bias_metrics['content_effect_inter_validity_label']:.2f}%")
    print(f"  Total Content Effect (acc_tot): {tot_content_effect:.2f}%")
    
    # --- 4. Combined Metric Calculation ---
    combined_smooth_score = calculate_smooth_combined_metric(overall_acc, tot_content_effect)
    
    print("\n" + "=" * 50)
    print("--- Combined Performance Scores ---")
    
    print(f"\n[2] Log-Smoothed Score (Accuracy / (1 + ln(1 + Bias))):")
    print(f"    Score: {combined_smooth_score:.2f}")
    print("=" * 50)

    return {
        'name': model_name,
        'overall_acc': overall_acc,
        'tot_content_effect': tot_content_effect,
        'combined_smooth_score': combined_smooth_score
    }


# --- 3. EXECUTION ---

if __name__ == "__main__":
    try:
        # Load data from the mock JSON strings
        ground_truth = json.loads(GROUND_TRUTH_DATA_JSON)
        gt_map = {item['id']: item for item in ground_truth}

        # Define the models to analyze
        models_to_analyze = {
            "MODEL 0 (Perfect Score Example)": json.loads(MOCK_PREDICTIONS_MODEL_0_JSON),
            "MODEL 1 (Higher Bias Example)": json.loads(MOCK_PREDICTIONS_MODEL_1_JSON),
            "MODEL 2 (Lower Bias Example)": json.loads(MOCK_PREDICTIONS_MODEL_2_JSON),
        }
        
        all_model_results = []

        # Step 1: Analyze models individually and store results
        for model_name, predictions in models_to_analyze.items():
            results = analyze_model_performance(model_name, ground_truth, predictions, gt_map)
            all_model_results.append(results)

        # Step 2: Print Comparison Table
        print("\n\n" + "#" * 70)
        print("--- FINAL RANKING AND COMPARATIVE SUMMARY ---")
        print("Based on the Log-Smoothed Score (Higher is Better: Max Acc, Min Bias)")
        print("#" * 70)

        # Sort by smooth Score (descending for ranking)
        sorted_results = sorted(all_model_results, key=lambda x: x['combined_smooth_score'], reverse=True)

        print("-" * 70)
        print(f"| {'Rank':<4} | {'Model Name':<25} | {'Overall Acc':>11} | {'Total Bias':>10} | {'Combined Score':>12} |")
        print("-" * 70)
        
        for rank, res in enumerate(sorted_results, 1):
            print(f"| {rank:<4} | {res['name']:<25} | {res['overall_acc']:10.2f}% | {res['tot_content_effect']:9.2f}% | {res['combined_smooth_score']:12.2f} |")
        
        print("-" * 70)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
