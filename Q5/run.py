
"""
Multilingual & Code-Switch Stress Test (Offline Mode, with Mitigation)
This script evaluates pre-collected model outputs for multilingual robustness.
Supports:
  - Accuracy, Fluency, Error analysis
  - Mini-intervention (language pinning system prompt + re-test on subset)
"""
import csv
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime


# ------------------- Data Structures -------------------

@dataclass
class PromptItem:
    id: int
    category: str
    english: str
    hindi: str
    hinglish: str
    code_switch: str
    gold_answer: str
    answer_type: str  # 'fact', 'number', 'yes_no', 'short_text'


@dataclass
class EvaluationResult:
    id: int
    condition: str
    prompt: str
    gold: str
    prediction: str
    correct: int
    fluency: int
    error_type: str


# ------------------- Stress Test Class -------------------

class OfflineStressTest:
    def __init__(self, predictions: Dict[Tuple[int, str], str]):
        """
        Args:
            predictions: dictionary mapping (id, condition) -> model output
                        conditions: 'L1'=English, 'L2'=Hindi,
                                    'L3'=Hinglish, 'CS'=Code-switch
        """
        self.predictions = predictions
        self.prompts = self._create_prompts()
        self.results: List[EvaluationResult] = []

    # ------------------- Prompt Creation -------------------

    def _create_prompts(self) -> List[PromptItem]:
            """Create 20 parallel prompts across 3 languages with code-switching variants"""
            prompts = [
                # Factual Questions (5 items)
                PromptItem(
                    id=1,
                    category="fact",
                    english="What is the capital of India?",
                    hindi="भारत की राजधानी क्या है?",
                    hinglish="India ki capital kya hai?",
                    code_switch="भारत की capital city kya है?",
                    gold_answer="New Delhi",
                    answer_type="fact"
                ),
                PromptItem(
                    id=2,
                    category="fact",
                    english="How many days are in a leap year?",
                    hindi="एक लीप वर्ष में कितने दिन होते हैं?",
                    hinglish="Leap year mein kitne days hote hain?",
                    code_switch="एक leap year में कितने days होते हैं?",
                    gold_answer="366 days",
                    answer_type="number"
                ),
                PromptItem(
                    id=3,
                    category="fact",
                    english="What color is the sky on a clear day?",
                    hindi="साफ़ दिन में आसमान का रंग कैसा होता है?",
                    hinglish="Clear day pe aasman ka color kaisa hota hai?",
                    code_switch="साफ़ day में sky का रंग कैसा होता है?",
                    gold_answer="blue",
                    answer_type="fact"
                ),
                PromptItem(
                    id=4,
                    category="fact",
                    english="Which planet is closest to the sun?",
                    hindi="सूर्य के सबसे निकट कौन सा ग्रह है?",
                    hinglish="Sun ke sabse pass kaunsa planet hai?",
                    code_switch="सूर्य के sabse close कौन सा planet है?",
                    gold_answer="Mercury",
                    answer_type="fact"
                ),
                PromptItem(
                    id=5,
                    category="fact",
                    english="How many continents are there?",
                    hindi="कितने महाद्वीप हैं?",
                    hinglish="Kitne continents hain?",
                    code_switch="कितने continents हैं world में?",
                    gold_answer="7",
                    answer_type="number"
                ),
                
                # Simple Math (5 items)
                PromptItem(
                    id=6,
                    category="math",
                    english="What is 15 plus 25?",
                    hindi="15 और 25 का योग क्या है?",
                    hinglish="15 plus 25 kitna hota hai?",
                    code_switch="15 plus 25 का sum क्या है?",
                    gold_answer="40",
                    answer_type="number"
                ),
                PromptItem(
                    id=7,
                    category="math",
                    english="What is half of 100?",
                    hindi="100 का आधा क्या है?",
                    hinglish="100 ka half kya hai?",
                    code_switch="100 का half कितना होता है?",
                    gold_answer="50",
                    answer_type="number"
                ),
                PromptItem(
                    id=8,
                    category="math",
                    english="What is 10 multiplied by 5?",
                    hindi="10 गुणा 5 कितना होता है?",
                    hinglish="10 multiply 5 kitna hota hai?",
                    code_switch="10 को 5 से multiply करने पर क्या मिलता है?",
                    gold_answer="50",
                    answer_type="number"
                ),
                PromptItem(
                    id=9,
                    category="math",
                    english="What is the square of 4?",
                    hindi="4 का वर्ग क्या है?",
                    hinglish="4 ka square kya hai?",
                    code_switch="4 का square कितना होता है?",
                    gold_answer="16",
                    answer_type="number"
                ),
                PromptItem(
                    id=10,
                    category="math",
                    english="What is 100 divided by 4?",
                    hindi="100 को 4 से भाग देने पर क्या मिलता है?",
                    hinglish="100 divided by 4 kitna hota hai?",
                    code_switch="100 को 4 से divide करने पर answer क्या है?",
                    gold_answer="25",
                    answer_type="number"
                ),
                
                # Yes/No Questions (5 items)
                PromptItem(
                    id=11,
                    category="yes_no",
                    english="Is water a liquid at room temperature?",
                    hindi="क्या कमरे के तापमान पर पानी तरल होता है?",
                    hinglish="Kya room temperature pe paani liquid hota hai?",
                    code_switch="क्या room temperature पर water liquid होता है?",
                    gold_answer="yes",
                    answer_type="yes_no"
                ),
                PromptItem(
                    id=12,
                    category="yes_no",
                    english="Can fish breathe air?",
                    hindi="क्या मछली हवा में सांस ले सकती है?",
                    hinglish="Kya fish air mein breathe kar sakti hai?",
                    code_switch="क्या fish हवा में breathe कर सकती है?",
                    gold_answer="no",
                    answer_type="yes_no"
                ),
                PromptItem(
                    id=13,
                    category="yes_no",
                    english="Is the sun a star?",
                    hindi="क्या सूर्य एक तारा है?",
                    hinglish="Kya sun ek star hai?",
                    code_switch="क्या sun एक star है?",
                    gold_answer="yes",
                    answer_type="yes_no"
                ),
                PromptItem(
                    id=14,
                    category="yes_no",
                    english="Do plants need sunlight?",
                    hindi="क्या पौधों को सूर्य की रोशनी चाहिए?",
                    hinglish="Kya plants ko sunlight chahiye?",
                    code_switch="क्या plants को sunlight की जरूरत होती है?",
                    gold_answer="yes",
                    answer_type="yes_no"
                ),
                PromptItem(
                    id=15,
                    category="yes_no",
                    english="Is ice heavier than water?",
                    hindi="क्या बर्फ पानी से भारी होती है?",
                    hinglish="Kya ice paani se heavy hoti hai?",
                    code_switch="क्या ice water से ज्यादा heavy होती है?",
                    gold_answer="no",
                    answer_type="yes_no"
                ),
                
                # Simple Instructions (5 items)
                PromptItem(
                    id=16,
                    category="instruction",
                    english="Name a fruit that is red.",
                    hindi="एक लाल रंग का फल बताइए।",
                    hinglish="Ek red color ka fruit batao.",
                    code_switch="एक red color का fruit बताइए।",
                    gold_answer="apple",  # or strawberry, cherry, etc.
                    answer_type="short_text"
                ),
                PromptItem(
                    id=17,
                    category="instruction",
                    english="Name the largest ocean.",
                    hindi="सबसे बड़ा महासागर बताइए।",
                    hinglish="Sabse bada ocean batao.",
                    code_switch="सबसे बड़ा ocean का नाम बताइए।",
                    gold_answer="Pacific",
                    answer_type="fact"
                ),
                PromptItem(
                    id=18,
                    category="instruction",
                    english="What animal is known as the king of the jungle?",
                    hindi="जंगल का राजा किस जानवर को कहा जाता है?",
                    hinglish="Jungle ka raja kis animal ko kehte hain?",
                    code_switch="जंगल का king किस animal को कहते हैं?",
                    gold_answer="lion",
                    answer_type="fact"
                ),
                PromptItem(
                    id=19,
                    category="instruction",
                    english="Name a primary color.",
                    hindi="एक प्राथमिक रंग बताइए।",
                    hinglish="Ek primary color batao.",
                    code_switch="एक primary color का नाम बताइए।",
                    gold_answer="red",  # or blue, yellow
                    answer_type="short_text"
                ),
                PromptItem(
                    id=20,
                    category="instruction",
                    english="What is the opposite of hot?",
                    hindi="गर्म का विपरीत क्या है?",
                    hinglish="Hot ka opposite kya hai?",
                    code_switch="Hot का opposite क्या होता है?",
                    gold_answer="cold",
                    answer_type="fact"
                )
            ]
            return prompts


    # ------------------- Evaluation Utilities -------------------

    def evaluate_response(self, gold: str, prediction: str, answer_type: str) -> int:
        """Check correctness of prediction against gold answer."""
        gold = gold.lower().strip()
        pred = (prediction or "").lower().strip()

        if answer_type == "number":
            import re
            g = re.findall(r"\d+", gold)
            p = re.findall(r"\d+", pred)
            return int(bool(g and p and g[0] == p[0]))

        elif answer_type == "yes_no":
            yes = ["yes", "हाँ", "haan", "ji", "true", "correct"]
            no = ["no", "नहीं", "nahi", "nahin", "false", "incorrect"]
            if gold in ["yes", "हाँ"]:
                return int(any(v in pred for v in yes) and not any(v in pred for v in no))
            else:
                return int(any(v in pred for v in no) and not any(v in pred for v in yes))

        elif answer_type == "fact":
            return int(gold in pred)

        else:  # short_text
            acceptable = {
                "apple": ["apple", "strawberry", "cherry", "tomato"],
                "red": ["red", "blue", "yellow"],
                "cold": ["cold", "cool", "chilly", "freezing"]
            }
            if gold in acceptable:
                return int(any(a in pred for a in acceptable[gold]))
            return int(gold in pred)

    def rate_fluency(self, response: str, condition: str) -> int:
        """Rate fluency on a 1–5 scale."""
        if not response:
            return 1
        score = 3
        if len(response.split()) > 3:
            score += 1
        if response[-1] in ".!?।":
            score += 1
        if "error" in response.lower() or "sorry" in response.lower():
            score -= 1
        return max(1, min(score, 5))

    def classify_error(self, prompt: str, gold: str, prediction: str) -> str:
        """Categorize error type."""
        if not prediction:
            return "no_response"
        pred = prediction.lower()
        if "sorry" in pred or "cannot" in pred:
            return "refusal"
        if len(pred) > len(gold) * 5:
            return "over_generation"
        return "incorrect_answer"

    # ------------------- Evaluation -------------------

    def run_evaluation(self, conditions: List[str] = None) -> List[EvaluationResult]:
        """Evaluate predictions across all prompts and conditions."""
        if conditions is None:
            conditions = ["L1", "L2", "L3", "CS"]

        condition_map = {"L1": "english", "L2": "hindi", "L3": "hinglish", "CS": "code_switch"}
        results = []

        for cond in conditions:
            for p in self.prompts:
                prompt_text = getattr(p, condition_map[cond])
                prediction = self.predictions.get((p.id, cond), "")

                correct = self.evaluate_response(p.gold_answer, prediction, p.answer_type)
                fluency = self.rate_fluency(prediction, cond)
                error = "" if correct else self.classify_error(prompt_text, p.gold_answer, prediction)

                results.append(EvaluationResult(
                    id=p.id, condition=cond, prompt=prompt_text,
                    gold=p.gold_answer, prediction=prediction,
                    correct=correct, fluency=fluency, error_type=error
                ))

        self.results = results
        return results

    def analyze_results(self):
        """Aggregate results into summary statistics."""
        if not self.results:
            return {}
        analysis = defaultdict(lambda: {"total": 0, "correct": 0, "fluency": [], "errors": defaultdict(int)})

        for r in self.results:
            a = analysis[r.condition]
            a["total"] += 1
            a["correct"] += r.correct
            a["fluency"].append(r.fluency)
            if r.error_type:
                a["errors"][r.error_type] += 1

        stats = {}
        for cond, d in analysis.items():
            acc = d["correct"] / d["total"]
            mean_f = np.mean(d["fluency"])
            n = d["total"]
            se = np.sqrt(acc * (1 - acc) / n)
            ci = (max(0, acc - 1.96 * se), min(1, acc + 1.96 * se))
            stats[cond] = {"accuracy": acc, "ci95": ci, "mean_fluency": mean_f, "n": n,
                           "errors": dict(d["errors"])}
        return stats

    def generate_report(self) -> str:
        """Generate a textual evaluation report."""
        stats = self.analyze_results()
        lines = ["="*50, "OFFLINE MULTILINGUAL STRESS TEST REPORT", "="*50,
                 f"Timestamp: {datetime.now().isoformat()}",
                 f"Conditions: {list(stats.keys())}"]

        for cond, d in stats.items():
            lines.append(f"\n{cond} ({d['n']} samples):")
            lines.append(f"  Accuracy: {d['accuracy']:.2%}")
            lines.append(f"  95% CI: [{d['ci95'][0]:.2%}, {d['ci95'][1]:.2%}]")
            lines.append(f"  Mean Fluency: {d['mean_fluency']:.2f}")
            if d["errors"]:
                lines.append("  Errors:")
                for et, c in d["errors"].items():
                    lines.append(f"    - {et}: {c}")
        return "\n".join(lines)

    # ------------------- Mitigation & Re-Test -------------------

    def re_test_subset(self, intervention_preds: Dict[Tuple[int, str], str]) -> Dict[str, Dict[str, float]]:
        """
        Compare baseline vs intervention on a 6-item subset.
        Args:
            intervention_preds: dict of (id, condition) -> output after intervention
        Returns:
            Dictionary of deltas (accuracy / fluency changes per condition).
        """
        subset_ids = [1, 2, 3, 4, 5, 6]
        baseline = [r for r in self.results if r.id in subset_ids]

        # Evaluate intervention subset
        inter_results = []
        for r in baseline:
            pred = intervention_preds.get((r.id, r.condition), "")
            correct = self.evaluate_response(r.gold, pred, "fact" if r.id in [1,3,4,5] else "number")
            fluency = self.rate_fluency(pred, r.condition)
            inter_results.append((r.condition, correct, fluency))

        # Aggregate baseline vs intervention
        deltas = defaultdict(lambda: {"baseline_acc": 0, "inter_acc": 0,
                                      "baseline_f": 0, "inter_f": 0, "n": 0})
        for r in baseline:
            d = deltas[r.condition]
            d["baseline_acc"] += r.correct
            d["baseline_f"] += r.fluency
            d["n"] += 1
        for cond, c, f in inter_results:
            d = deltas[cond]
            d["inter_acc"] += c
            d["inter_f"] += f

        # Normalize and compute deltas
        out = {}
        for cond, d in deltas.items():
            n = d["n"]
            out[cond] = {
                "baseline_acc": d["baseline_acc"]/n,
                "inter_acc": d["inter_acc"]/n,
                "delta_acc": (d["inter_acc"] - d["baseline_acc"]) / n,
                "baseline_f": d["baseline_f"]/n,
                "inter_f": d["inter_f"]/n,
                "delta_f": (d["inter_f"] - d["baseline_f"]) / n
            }
        return out


# ------------------- Example Usage -------------------

if __name__ == "__main__":
    # Example: paste baseline predictions here
    predictions = {
        (1, "L1"): "New Delhi",
        (1, "L2"): "नई दिल्ली",
        (1, "L3"): "नई दिल्ली",
        (1, "CS"): "नई दिल्ली",
        (2, "L1"): "366 days",
        (2, "L2"): "365 दिन",
        (2, "L3"): "366 दिन",
        (2, "CS"): "366 दिन",
        (3, "L1"): "blue",
        (3, "L2"): "blue",
        (3, "L3"): "blue",
        (3, "CS"): "red",
        (4, "L1"): "Mercury",
        (4, "L2"): "Mercury",
        (4, "L3"): "Mercury",
        (4, "CS"): "Mercury",
        (5, "L1"): "7",
        (5, "L2"): "7", 
        (5, "L3"): "7",
        (5, "CS"): "7", 
        (6, "L1"): "40",
        (6, "L2"): "40",
        (6, "L3"): "40",
        (6, "CS"): "40",
        (7, "L1"): "50",
        (7, "L2"): "50",
        (7, "L3"): "50",
        (7, "CS"): "50",
        (8, "L1"): "50",
        (8, "L2"): "50",
        (8, "L3"): "50",
        (8, "CS"): "50",
        (9, "L1"): "16",
        (9, "L2"): "16",
        (9, "L3"): "16",
        (9, "CS"): "16",
        (10, "L1"): "25",
        (10, "L2"): "25",
        (10, "L3"): "25",
        (10, "CS"): "25",
        (11, "L1"): "yes",
        (11, "L2"): "हाँ",
        (11, "L3"): "हाँ",
        (11, "CS"): "yes",
        (12, "L1"): "no",
        (12, "L2"): "नहीं",
        (12, "L3"): "नहीं",
        (12, "CS"): "yes",
        (13, "L1"): "yes",
        (13, "L2"): "हाँ",
        (13, "L3"): "हाँ",
        (13, "CS"): "yes",
        (14, "L1"): "yes",
        (14, "L2"): "हाँ",
        (14, "L3"): "हाँ",
        (14, "CS"): "yes",
        (15, "L1"): "no",
        (15, "L2"): "नहीं",
        (15, "L3"): "नहीं",
        (15, "CS"): "no",
        (16, "L1"): "apple",
        (16, "L2"): "सेब",
        (16, "L3"): "सेब",
        (16, "CS"): "apple",
        (17, "L1"): "Pacific Ocean",
        (17, "L2"): "प्रशांत महासागर",
        (17, "L3"): "प्रशांत महासागर",
        (17, "CS"): "प्रशांत महासागर",
        (18, "L1"): "lion",
        (18, "L2"): "शेर",
        (18, "L3"): "शेर",
        (18, "CS"): "lion",
        (19, "L1"): "red",
        (19, "L2"): "लाल",
        (19, "L3"): "लाल",
        (19, "CS"): "लाल",
        (20, "L1"): "cold",
        (20, "L2"): "ठंडा",
        (20, "L3"): "ठंडा",
        (20, "CS"): "cold",
    }

    tester = OfflineStressTest(predictions)
    tester.run_evaluation()
    print(tester.generate_report())


    intervention_preds = {
        (1, "L1"): "नई दिल्ली",
        (1, "L2"): "नई दिल्ली",
        (1, "L3"): "नई दिल्ली",
        (1, "CS"): "नई दिल्ली",
        (2, "L1"): "366 दिन",
        (2, "L2"): "366 दिन",
        (2, "L3"): "366 दिन",
        (2, "CS"): "366 दिन",
        (3, "L1"): "नीला",
        (3, "L2"): "नीला",
        (3, "L3"): "नीला",
        (3, "CS"): "नीला",
        (4, "L1"): "बुध",
        (4, "L2"): "बुध",
        (4, "L3"): "बुध",
        (4, "CS"): "बुध",
        (5, "L1"): "7",
        (5, "L2"): "7",
        (5, "L3"): "7",
        (5, "CS"): "7",
        (6, "L1"): "40",
        (6, "L2"): "40",
        (6, "L3"): "40",
        (6, "CS"): "40"
    }

    deltas = tester.re_test_subset(intervention_preds)
    print("\n--- DELTA REPORT (After Intervention) ---")
    for cond, d in deltas.items():
        print(f"{cond}: ΔAcc={d['delta_acc']:.2f}, ΔFluency={d['delta_f']:.2f}")
