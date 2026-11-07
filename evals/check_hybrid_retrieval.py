"""Read-only smoke TASK-3.2: embedding + retrieval, без генераций и Telegram."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from vsitemssrag.embeddings import EMBEDDING_MODEL, GeminiEmbeddingClient, build_search_query
from vsitemssrag.rag.constraints import extract_max_price_rub
from vsitemssrag.rag.materials import material_context
from vsitemssrag.rag.popularity import bestseller
from vsitemssrag.rag.reranker import RETRIEVAL_VERSION, candidate_pool_limit, select_ranked_products
from vsitemssrag.settings import require_setting
from vsitemssrag.storage.postgres import Database


CASES = (
    ("beginner_control", "Я новичок, хочу контрольное основание"),
    ("beginner_budget", "Я новичок, бюджет до 6000 ₽, хочу контрольное основание"),
    ("explicit_dotec", "Я новичок, хочу Donic Dotec с пробковой ручкой"),
    ("attack_carbon", "Ищу быстрое карбоновое основание для атакующей игры"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/exports/hybrid_retrieval_smoke.json"))
    parser.add_argument("--case", choices=[case[0] for case in CASES], action="append")
    args = parser.parse_args()
    selected = [case for case in CASES if not args.case or case[0] in args.case]
    with GeminiEmbeddingClient(require_setting("GEMINI_API_KEY"), delay=1, timeout=30) as client:
        vectors = [client.embed(build_search_query(question)) for _, question in selected]
    cases = []
    with Database.from_env() as db:
        for (case_id, question), vector in zip(selected, vectors, strict=True):
            pool = db.match_products(vector, model=EMBEDDING_MODEL, limit=candidate_pool_limit(6), only_available=True)
            budget = extract_max_price_rub(question)
            ranked = select_ranked_products(question, pool, budget, limit=6)
            checks = {
                "nonempty": bool(ranked),
                "budget": all(p.get("price") is not None and 0 < Decimal(str(p["price"])) <= budget for p in ranked) if budget else True,
                "sorted_scores": [p['ranking']['final_score'] for p in ranked] == sorted((p['ranking']['final_score'] for p in ranked), reverse=True),
            }
            if case_id.startswith("beginner"):
                checks["control_classic_top3"] = any((entry := bestseller(p)) is not None and entry.category == 'control' for p in ranked[:3])
                checks["no_dotec_top3"] = not any('dotec' in p['name'].casefold() for p in ranked[:3])
            if case_id == 'explicit_dotec':
                checks['no_niche_penalty'] = all(p['ranking']['niche_penalty'] == 0 for p in ranked)
            report = {
                'case_id': case_id, 'question': question, 'checks': checks,
                'semantic_pool_size': len(pool),
                'candidates': pool,
                'semantic_top3': [p['name'] for p in pool[:3]],
                'ranked': [{key: p.get(key) for key in ('product_id', 'name', 'price', 'ranking')} |
                           {'material_reference': material_context(p)} for p in ranked],
            }
            cases.append(report)
            print(case_id + ': ' + ', '.join(p['name'] for p in ranked[:3]), flush=True)
            print('  ' + str(checks), flush=True)
    root = Path(__file__).resolve().parents[1] / 'src' / 'vsitemssrag'
    fingerprints = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ('rag/popularity.py', 'rag/reranker.py', 'rag/materials.py', 'rag/materials_ontology.json', 'storage/postgres.py')}
    payload = {'timestamp': datetime.now(UTC).isoformat(), 'retrieval_version': RETRIEVAL_VERSION,
               'code_sha256': fingerprints,
               'generation_calls': 0, 'cases': cases, 'passed': all(all(c['checks'].values()) for c in cases)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print(f'Report: {args.output}', flush=True)
    return 0 if payload['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
