import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from backend.llm import llm as rag_llm
from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper

load_dotenv()

PDF_PATH            = "Openclaw_Research_Report.pdf"
GOLDENS_FILE        = Path("goldens.json")
MAX_CONTEXTS        = 5
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7


class EvaluationLLM(DeepEvalBaseLLM):
    def __init__(self) -> None:
        super().__init__(model=rag_llm.model)

    def load_model(self) -> BaseChatModel:
        return rag_llm

    def generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> str | BaseModel:
        model = self.load_model()
        response = (
            model.with_structured_output(schema).invoke(prompt)
            if schema is not None
            else model.invoke(prompt)
        )
        if schema is not None:
            return response
        return response.content

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> str | BaseModel:
        model = self.load_model()
        response = (
            await model.with_structured_output(schema).ainvoke(prompt)
            if schema is not None
            else await model.ainvoke(prompt)
        )
        if schema is not None:
            return response
        return response.content

    def get_model_name(self) -> str:
        return rag_llm.model


eval_llm = EvaluationLLM()


def generate_goldens() -> list[dict]:
    synthesizer = Synthesizer(model=eval_llm)
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[PDF_PATH],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS,
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return pairs


def load_goldens() -> list[dict]:
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))


def run_rag_query(
    graph, query: str, session_id: str, thread_id: str
) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation in resumable batches.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of golden questions to evaluate at a time (default: 1).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based golden-question index to start from.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Exclusive zero-based golden-question index to stop at.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens()
    if args.start_index >= len(pairs):
        parser.error(f"--start-index must be less than the number of goldens ({len(pairs)})")
    end_index = args.end_index if args.end_index is not None else len(pairs)
    if end_index > len(pairs):
        parser.error(f"--end-index cannot exceed the number of goldens ({len(pairs)})")
    if end_index <= args.start_index:
        parser.error("--end-index must be greater than --start-index")

    docs = load_document(PDF_PATH)
    graph = build_graph(db_path="eval_checkpoints.db")

    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=eval_llm),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=eval_llm),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=eval_llm),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=eval_llm),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=eval_llm),
    ]

    results_path = Path("eval_results.json")
    summary = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else []
    )
    completed_indices = {
        item["case_index"]
        for item in summary
        if item.get("status") == "complete" and isinstance(item.get("case_index"), int)
    }
    session_id = f"evaluation_session_{uuid4()}"
    add_paper(docs, session_id)

    indices = [
        index
        for index in range(args.start_index, end_index)
        if index not in completed_indices
    ]
    if not indices:
        print("All requested evaluation cases are already complete.")
        return

    def save_results() -> None:
        temporary_path = results_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(results_path)

    for offset in range(0, len(indices), args.batch_size):
        batch_indices = indices[offset : offset + args.batch_size]
        test_cases = []
        case_indices = []
        for index in batch_indices:
            pair = pairs[index]
            query = pair["input"] + " as per the report in knowledge base"
            try:
                answer, retrieval_context = run_rag_query(
                    graph,
                    query,
                    session_id,
                    thread_id=f"{session_id}_{index}",
                )
            except Exception as error:
                summary[:] = [item for item in summary if item.get("case_index") != index]
                summary.append({
                    "case_index": index,
                    "input": pair["input"],
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                })
                save_results()
                print(f"Case {index + 1}/{len(pairs)} failed during RAG query: {error}")
                continue

            test_cases.append(
                LLMTestCase(
                    input=pair["input"],
                    actual_output=answer,
                    expected_output=pair["expected_output"],
                    retrieval_context=retrieval_context,
                )
            )
            case_indices.append(index)

        if not test_cases:
            continue

        try:
            results = evaluate(
                test_cases,
                metrics,
                async_config=AsyncConfig(max_concurrent=1, throttle_value=5),
            )
        except Exception as error:
            for index, test_case in zip(case_indices, test_cases):
                summary[:] = [item for item in summary if item.get("case_index") != index]
                summary.append({
                    "case_index": index,
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                })
            save_results()
            print(f"Batch starting at case {batch_indices[0] + 1} failed during metrics: {error}")
            continue

        if len(results.test_results) != len(case_indices):
            raise RuntimeError(
                "DeepEval returned a different number of results than test cases; "
                "refusing to associate results with the wrong questions."
            )
        for index, test_result in zip(case_indices, results.test_results):
            summary[:] = [item for item in summary if item.get("case_index") != index]
            summary.append({
                "case_index": index,
                "input": test_result.input,
                "actual_output": test_result.actual_output,
                "status": "complete",
                "success": test_result.success,
                "metrics": [
                    {
                        "name": metric.name,
                        "score": metric.score,
                        "passed": metric.success,
                        "reason": metric.reason,
                    }
                    for metric in test_result.metrics_data
                ],
            })
        save_results()
        print(
            f"Completed cases {batch_indices[0] + 1}-{batch_indices[-1] + 1} "
            f"of {len(pairs)}. Results saved to {results_path}."
        )

    failed_count = sum(item.get("status") == "error" for item in summary)
    print(f"\nEvaluation run finished with {failed_count} failed case(s). Results: {results_path}.")


if __name__ == "__main__":
    main()
