
import argparse
import json

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from rag_pipeline import DocumentQA


def run(engine, questions, baseline):
    rows = []
    for q in questions:
        answer, docs = engine.answer(q, baseline=baseline)
        rows.append({
            "user_input": q,
            "response": answer,
            "retrieved_contexts": [d.page_content for d in docs],
        })
    return rows


def score(engine, rows):
    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(engine.llm),
        embeddings=LangchainEmbeddingsWrapper(engine.embeddings),
    )
    df = result.to_pandas()
    return float(df["faithfulness"].mean()), float(df["answer_relevancy"].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="+", required=True)
    parser.add_argument("--questions", required=True)
    args = parser.parse_args()

    questions = [item["question"] for item in json.load(open(args.questions))]
    engine = DocumentQA()
    engine.add_files([(p, None) for p in args.docs])

    results = {}
    for name, baseline in [("baseline", True), ("full pipeline", False)]:
        print(f"Evaluating {name} on {len(questions)} questions...")
        results[name] = score(engine, run(engine, questions, baseline))

    print("\n=== RAGAS results ===")
    for name, (faith, rel) in results.items():
        print(f"{name:14s} faithfulness={faith:.3f}  hallucination_rate={1 - faith:.3f}  "
              f"answer_relevancy={rel:.3f}")

    base_h = 1 - results["baseline"][0]
    full_h = 1 - results["full pipeline"][0]
    if base_h > 0:
        print(f"\nRelative hallucination reduction: {(base_h - full_h) / base_h:.0%}")


if __name__ == "__main__":
    main()
