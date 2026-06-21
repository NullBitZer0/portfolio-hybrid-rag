"""RAGAS evaluation script for the Hybrid RAG system."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import ingest
from src.retrieval import get_reranked_retriever
from src.pipeline import build_conversational_rag_chain, ConversationMemory
from src.config import GROQ_API_KEY, LLM_MODEL


def load_golden_dataset(path: str = "evals/golden_dataset.json") -> list:
    with open(path) as f:
        return json.load(f)


def run_rag(chain, question: str) -> dict:
    result = chain(question)
    return result


def build_evaluation_dataset(golden: list, chain, retriever) -> list:
    """Run RAG on each question and build RAGAS evaluation dataset."""
    dataset = []

    for item in golden:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  Evaluating: {question[:60]}...")

        result = chain(question)
        answer = result["answer"]

        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        dataset.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": ground_truth,
        })

    return dataset


def evaluate(dataset: list) -> dict:
    """Run RAGAS evaluation on the dataset."""
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
    )
    from langchain_groq import ChatGroq

    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.1)
    )

    from ragas.dataset_schema import EvaluationDataset

    eval_dataset = EvaluationDataset.from_list(dataset)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
    )

    return result


def main():
    print("=" * 60)
    print("  RAGAS Evaluation")
    print("=" * 60)

    print("\n[1/4] Loading golden dataset...")
    golden = load_golden_dataset()
    print(f"  Loaded {len(golden)} test cases")

    print("\n[2/4] Initializing RAG system...")
    total_chunks = ingest()
    retriever = get_reranked_retriever()
    memory = ConversationMemory()
    chain = build_conversational_rag_chain(retriever, memory)

    print("\n[3/4] Running RAG on test cases...")
    dataset = build_evaluation_dataset(golden, chain, retriever)
    print(f"  Built {len(dataset)} evaluation samples")

    print("\n[4/4] Running RAGAS evaluation...")
    result = evaluate(dataset)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    scores = result.scores
    for metric, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:25s} {bar} {score:.3f}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  {'AVERAGE':25s} {'█' * int(avg * 20)}{'░' * (20 - int(avg * 20))} {avg:.3f}")

    output_path = "evals/evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return result


if __name__ == "__main__":
    main()
