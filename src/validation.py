import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from src.config import VALIDATION_REPORT_FILE
from src.logger import logger

@dataclass
class ValidationItemResult:
    question_id: str
    is_valid: bool
    checks: Dict[str, bool]
    details: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "is_valid": self.is_valid,
            "checks": self.checks,
            "details": self.details
        }

class CitationValidator:
    def __init__(self):
        pass

    def validate_record(self, answer_record: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> ValidationItemResult:
        """
        Deterministically validates an answer record against its retrieved context.
        Enforces:
        1. Non-empty answer
        2. Every supported answer includes at least one citation
        3. Cited documents must appear in the retrieved chunk set
        4. Unsupported answers must have empty citations and supported=False
        """
        q_id = answer_record.get("question_id", "unknown")
        answer = answer_record.get("answer", "").strip()
        citations = answer_record.get("citations", [])
        supported = bool(answer_record.get("supported", False))

        retrieved_doc_ids = {c.get("doc_id") for c in retrieved_chunks}

        checks = {
            "non_empty_answer": len(answer) > 0,
            "citations_present_if_supported": True,
            "citations_in_retrieved_docs": True,
            "supported_flag_consistent": True
        }
        details = []

        # Check 1: Non-empty answer
        if not checks["non_empty_answer"]:
            details.append("Answer text is empty.")

        # Check 2: If supported, must have at least one citation
        if supported:
            if not citations:
                checks["citations_present_if_supported"] = False
                details.append("Supported answer is missing citations.")

        # Check 3: Cited documents must exist in retrieved chunks
        invalid_citations = [doc for doc in citations if doc not in retrieved_doc_ids]
        if invalid_citations:
            checks["citations_in_retrieved_docs"] = False
            details.append(f"Citations not present in retrieved context: {invalid_citations}")

        # Check 4: Unsupported answers shouldn't claim citations
        if not supported and citations:
            checks["supported_flag_consistent"] = False
            details.append("Unsupported answer includes citations.")

        # Overall validity: all active checks must pass
        is_valid = all(checks.values())
        if is_valid:
            details.append("All deterministic validation checks passed.")

        logger.info(f"[VALIDATION] QID: {q_id} -> Valid: {is_valid} ({', '.join(details)})")
        return ValidationItemResult(
            question_id=q_id,
            is_valid=is_valid,
            checks=checks,
            details=details
        )

    def validate_all(
        self,
        answers: List[Dict[str, Any]],
        retrieval_results: List[Dict[str, Any]],
        output_file: Path = VALIDATION_REPORT_FILE
    ) -> Dict[str, Any]:
        """
        Runs deterministic validation across all answers and generates validation_report.json.
        """
        # Map question_id -> retrieved_chunks
        retrieval_map = {r["question_id"]: r.get("retrieved_chunks", []) for r in retrieval_results}

        item_results = []
        for ans in answers:
            q_id = ans["question_id"]
            retrieved = retrieval_map.get(q_id, [])
            item_res = self.validate_record(ans, retrieved)
            item_results.append(item_res)

        total = len(item_results)
        passed = sum(1 for r in item_results if r.is_valid)
        report = {
            "summary": {
                "total_evaluated": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": round((passed / total * 100), 2) if total > 0 else 0.0
            },
            "results": [r.to_dict() for r in item_results]
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"[VALIDATION] Report saved to {output_file} (Pass rate: {report['summary']['pass_rate']}%)")
        return report
