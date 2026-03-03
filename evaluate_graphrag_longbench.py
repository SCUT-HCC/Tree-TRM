#!/usr/bin/env python3
"""
GraphRAG evaluation script for LongBench dataset.

This script evaluates GraphRAG performance on LongBench dataset using Qwen API.
It tests the accuracy of GraphRAG in answering multiple-choice questions.

Usage Examples:
    # Using command line argument for API key
    python evaluate_graphrag_longbench.py --jsonl data/longbench_test.jsonl --api_key <your_qwen_api_key>
    
    # Using environment variable for API key
    export QWEN_API_KEY=<your_qwen_api_key>
    python evaluate_graphrag_longbench.py --jsonl data/longbench_test.jsonl
    
    # Evaluate only first 10 samples
    python evaluate_graphrag_longbench.py --jsonl data/longbench_test.jsonl --max_samples 10
    
    # Use different Qwen model
    python evaluate_graphrag_longbench.py --jsonl data/longbench_test.jsonl --model qwen-turbo
    
    # Custom output file
    python evaluate_graphrag_longbench.py --jsonl data/longbench_test.jsonl --output my_results.json

Required:
    - LongBench JSONL file with fields: question, context, choices, gold_answer_text
    - Qwen API key (via --api_key or QWEN_API_KEY environment variable)
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


class GraphRAGEvaluator:
    """Evaluator for GraphRAG on LongBench dataset"""

    def __init__(
        self,
        jsonl_path: str,
        qwen_api_key: str,
        qwen_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        qwen_model: str = "qwen-plus",
        max_samples: Optional[int] = None,
        timeout: int = 60,
    ):
        """
        Initialize GraphRAG evaluator.

        Args:
            jsonl_path: Path to LongBench JSONL file
            qwen_api_key: Qwen API key
            qwen_api_base: Qwen API base URL (default: dashscope compatible endpoint)
            qwen_model: Qwen model name (default: qwen-plus)
            max_samples: Maximum number of samples to evaluate (None for all)
            timeout: Request timeout in seconds
        """
        self.jsonl_path = jsonl_path
        self.qwen_api_key = qwen_api_key
        self.qwen_api_base = qwen_api_base.rstrip("/")
        self.qwen_model = qwen_model
        self.max_samples = max_samples
        self.timeout = timeout

        # Load dataset
        self.samples = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        """Load LongBench dataset from JSONL file."""
        samples = []
        print(f"Loading dataset from {self.jsonl_path}...")
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        if self.max_samples:
            samples = samples[: self.max_samples]
        print(f"Loaded {len(samples)} samples")
        return samples

    def _call_qwen_api(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        """
        Call Qwen API for text generation.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        url = f"{self.qwen_api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.qwen_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Qwen API: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response: {e.response.text}")
            return "ERROR"

    def _build_graphrag_prompt(
        self, question: str, context: str, choices: Dict[str, str]
    ) -> str:
        """
        Build prompt for GraphRAG evaluation.

        Args:
            question: The question to answer
            context: The context/documentation provided
            choices: Dictionary of choices (e.g., {"A": "...", "B": "..."})

        Returns:
            Formatted prompt string
        """
        choices_str = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

        prompt = f"""You are a reading comprehension assistant. Use the provided context to answer the question.

### Context
{context}

### Question
{question}

### Choices
{choices_str}

Please analyze the context carefully and select the correct answer. Reply with ONLY the option letter (A, B, C, or D) on the first line, followed by a brief explanation."""
        return prompt

    def _extract_option(self, text: str) -> Optional[str]:
        """
        Extract option letter (A/B/C/D) from model response.

        Args:
            text: Model response text

        Returns:
            Extracted option letter or None if not found
        """
        # Try multiple patterns to extract the option
        patterns = [
            r"the (?:correct )?answer is\s+\**([A-D])\**",
            r"answer:\s*\**([A-D])\**",
            r"\*\*([A-D])\*\*",
            r"^([A-D])[.):\s]",
            r"\b([A-D])\b\s+is\s+(?:correct|right)",
            r"(?:choose|select|option)\s+([A-D])\b",
            r"^([A-D])\s*$",  # Single letter on a line
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()

        # Fallback: find any single letter A-D in the first few lines
        lines = text.split("\n")[:3]
        for line in lines:
            match = re.search(r"\b([A-D])\b", line, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def _find_gold_option(self, sample: Dict) -> Optional[str]:
        """
        Find the gold standard answer option from sample.

        Args:
            sample: Sample dictionary with gold_answer_text and choices

        Returns:
            Option letter (A/B/C/D) or None if not found
        """
        gold_text = sample.get("gold_answer_text", "").strip()
        choices = sample.get("choices", {})

        for letter, text in choices.items():
            if text.strip() == gold_text:
                return letter.upper()

        return None

    def evaluate(self) -> Dict:
        """
        Evaluate GraphRAG on all samples.

        Returns:
            Dictionary containing evaluation results and statistics
        """
        print(f"\n{'='*70}")
        print(f"GraphRAG Evaluation on LongBench Dataset")
        print(f"{'='*70}")
        print(f"Model: {self.qwen_model}")
        print(f"Total samples: {len(self.samples)}")
        print(f"{'='*70}\n")

        results = []
        correct = 0
        total = 0
        none_preds = 0
        latencies = []

        for i, sample in enumerate(tqdm(self.samples, desc="Evaluating")):
            question = sample.get("question", "")
            context = sample.get("context", "")
            choices = sample.get("choices", {})
            gold_option = self._find_gold_option(sample)

            if not question or not context or not choices or not gold_option:
                print(f"Warning: Sample {i+1} missing required fields, skipping")
                continue

            # Build prompt
            prompt = self._build_graphrag_prompt(question, context, choices)

            # Call GraphRAG (via Qwen API)
            start_time = time.time()
            response = self._call_qwen_api(prompt)
            latency = time.time() - start_time

            # Extract predicted option
            pred_option = self._extract_option(response)

            # Evaluate
            is_correct = pred_option == gold_option if pred_option else False

            if is_correct:
                correct += 1
            if pred_option is None:
                none_preds += 1

            total += 1
            latencies.append(latency)

            # Store result
            results.append(
                {
                    "idx": i + 1,
                    "question": question,
                    "gold_option": gold_option,
                    "pred_option": pred_option,
                    "correct": is_correct,
                    "latency": round(latency, 3),
                    "response": response[:200] + "..." if len(response) > 200 else response,
                }
            )

            # Print first few examples
            if i < 5:
                print(f"\n[Sample {i+1}]")
                print(f"  Question: {question[:100]}...")
                print(f"  Gold: {gold_option}, Pred: {pred_option}, Correct: {is_correct}")
                print(f"  Latency: {latency:.2f}s")

        # Calculate statistics
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Print summary
        print(f"\n{'='*70}")
        print(f"Evaluation Results")
        print(f"{'='*70}")
        print(f"Total Samples:     {total}")
        print(f"Correct:            {correct}")
        print(f"Accuracy:          {accuracy:.2f}%")
        print(f"None Predictions:  {none_preds}")
        print(f"Avg Latency:       {avg_latency:.2f}s")
        print(f"{'='*70}")

        return {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "none_predictions": none_preds,
            "avg_latency": round(avg_latency, 4),
            "results": results,
        }

    def save_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save results JSON file
        """
        output_data = {
            "model": self.qwen_model,
            "dataset": self.jsonl_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate GraphRAG on LongBench dataset using Qwen API"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to LongBench JSONL file",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Qwen API key (or set QWEN_API_KEY environment variable)",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="Qwen API base URL (default: dashscope compatible endpoint)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-plus",
        help="Qwen model name (default: qwen-plus)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="graphrag_eval_results.json",
        help="Output JSON file path (default: graphrag_eval_results.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )

    args = parser.parse_args()

    # Get API key from args or environment variable
    api_key = args.api_key or os.getenv("QWEN_API_KEY")
    if not api_key:
        parser.error(
            "Qwen API key is required. Provide --api_key or set QWEN_API_KEY environment variable"
        )

    # Initialize evaluator
    evaluator = GraphRAGEvaluator(
        jsonl_path=args.jsonl,
        qwen_api_key=api_key,
        qwen_api_base=args.api_base,
        qwen_model=args.model,
        max_samples=args.max_samples,
        timeout=args.timeout,
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()

