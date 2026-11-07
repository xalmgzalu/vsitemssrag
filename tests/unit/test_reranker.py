"""Ранжирование, сохранение ограничений и достоверность материалов без сети."""
import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evals.run_enrichment_ab import DEFAULT_OUTPUT, DEFAULT_PROTOCOL, _configuration, _load_checkpoint, _retrieve_cases
from vsitemssrag.rag.constraints import requires_clarification
from vsitemssrag.rag.context import build_rag_prompt
from vsitemssrag.rag.materials import material_context, material_facts
from vsitemssrag.rag.popularity import BESTSELLERS, bestseller, ranking_modifiers
from vsitemssrag.rag.reranker import candidate_pool_limit, select_ranked_products
from vsitemssrag.rag.service import ask_catalog


def blade(name, similarity=0.7, price=4000, **extra):
    return {"name": name, "similarity": similarity, "price": price, **extra}


class RerankerTests(unittest.TestCase):
    def test_beginner_top_three_contains_classics_and_dotec_is_penalized(self):
        candidates = [blade("DONIC Original Dotec ALL", .92), blade("DONIC Appelgren Dotec Control", .91)]
        candidates += [blade(f"Обычное основание {i}", .84) for i in range(6)]
        candidates += [blade("DONIC Appelgren Allplay", .72), blade("STIGA Allround Classic", .71)]
        result = select_ranked_products("Я новичок, хочу контрольное основание", candidates, None, limit=3)
        self.assertEqual([p['name'] for p in result[:2]], ["DONIC Appelgren Allplay", "STIGA Allround Classic"])
        self.assertFalse(any("Dotec" in p['name'] for p in result))
        boost, penalty = ranking_modifiers("Новичок, контроль", candidates[0])
        self.assertEqual((boost, penalty), (0, .25))

    def test_budget_precedes_ranking_and_context_truncation(self):
        candidates = [blade("DONIC Appelgren Allplay", .9, 6001), blade("STIGA Allround Classic", .8, None)]
        candidates += [blade(f"Обычное {i}", .8, 5000) for i in range(6)]
        candidates += [blade("YASAKA Sweden Extra", .7, Decimal("6000.00"))]
        result = select_ranked_products("Новичок, до 6000", candidates, Decimal(6000), limit=3)
        self.assertEqual(result[0]['name'], "YASAKA Sweden Extra")
        self.assertTrue(all(p['price'] is not None and 0 < Decimal(str(p['price'])) <= 6000 for p in result))

    def test_popularity_cannot_rescue_over_budget_unknown_or_invalid_price(self):
        for price in (None, "NaN", "Infinity", "bad", -1, 0, 6001):
            with self.subTest(price=price):
                result = select_ranked_products("Новичок до 6000", [blade("DONIC Appelgren Allplay", 1, price)], Decimal(6000), limit=3)
                self.assertEqual(result, [])

    def test_niche_penalty_is_removed_for_requested_handle_or_dotec(self):
        item = blade("DONIC Original Dotec ALL")
        for query in ("Новичок, хочу Dotec", "Новичок, пробковая ручка", "Новичок, анатомическая ручка", "Новичок, эргономичная ручка"):
            with self.subTest(query=query):
                self.assertEqual(ranking_modifiers(query, item)[1], 0)
        self.assertEqual(ranking_modifiers("Новичок, без пробковой ручки", item)[1], .25)
        self.assertEqual(ranking_modifiers("Новичок, не хочу Dotec", item)[1], .25)

    def test_normal_anatomic_handle_is_not_a_niche_construction(self):
        item = blade("DONIC Appelgren Allplay", characteristics={"Ручка": "Анатомическая AN"})
        self.assertEqual(ranking_modifiers("Новичок", item), (.25, 0))

    def test_attack_and_defense_and_specialized_material_queries_keep_intent(self):
        attack = blade("BUTTERFLY Viscaria")
        control = blade("DONIC Appelgren Allplay")
        self.assertEqual(ranking_modifiers("Для атаки OFF", attack), (.2, 0))
        self.assertEqual(ranking_modifiers("Для атаки OFF", control), (0, 0))
        self.assertEqual(ranking_modifiers("Новичок, контроль", attack), (0, 0))
        self.assertEqual(ranking_modifiers("Защита под длинные шипы", control), (0, 0))
        self.assertEqual(ranking_modifiers("Хочу контрольное inner carbon", control), (0, 0))
        self.assertEqual(ranking_modifiers("Новичок, хочу Dotec", control), (0, 0))

    def test_registry_matches_models_not_substrings_or_unrelated_modifications(self):
        for name in ("Основание DONIC Appelgren Allplay FL", "BUTTERFLY Zoran Primorac", "BUTTERFLY Petr Korbel (JP)"):
            self.assertIsNotNone(bestseller({"name": name}), name)
        for name in ("DONIC Appelgren Dotec Control", "DONIC Appelgren Exclusive AR", "BUTTERFLY Primorac Carbon Tamca T5000", "BUTTERFLY Viscaria Super ALC", "STIGA Cybershape Carbon CWT Truls Edition", "BUTTERFLY Petr Korbel SK7"):
            self.assertIsNone(bestseller({"name": name}), name)
        self.assertTrue(all(.15 <= entry.popularity_boost <= .25 for entry in BESTSELLERS))

    def test_popularity_does_not_promote_wood_into_carbon_or_outer_into_inner(self):
        wood = blade('BUTTERFLY Petr Korbel', .8, characteristics={'Конструкция': 'Без композита'})
        carbon = blade('STIGA Cybershape Carbon', .7, characteristics={'Конструкция': 'Inner Carbon'})
        self.assertEqual(ranking_modifiers('Атака, карбон', wood)[0], 0)
        self.assertEqual(ranking_modifiers('Атака, inner carbon', carbon)[0], .2)
        self.assertEqual(ranking_modifiers('Атака, outer carbon', carbon)[0], 0)
        self.assertEqual(ranking_modifiers('Контроль, без композита', blade('DONIC Appelgren Allplay', characteristics={'Конструкция': 'Без композита'}))[0], .25)

    def test_attack_with_control_does_not_turn_into_beginner_intent(self):
        self.assertEqual(ranking_modifiers('Я не новичок, нужна атака с контролем', blade('BUTTERFLY Petr Korbel'))[0], .2)
        self.assertEqual(ranking_modifiers('Я не новичок, нужна атака с контролем', blade('DONIC Appelgren Allplay'))[0], 0)

    def test_scores_are_finite_ties_stable_and_candidates_are_not_mutated(self):
        candidates = [blade("A", .75), blade("B", .75)]
        original = copy.deepcopy(candidates)
        first = select_ranked_products("Основание", candidates, None, limit=6)
        self.assertEqual([p['name'] for p in first], ["A", "B"])
        self.assertEqual(candidates, original)
        self.assertEqual(first, select_ranked_products("Основание", candidates, None, limit=6))
        for value in (None, "bad", float('nan'), float('inf'), 10, True):
            item = select_ranked_products("Основание", [blade("A", value)], None, limit=1)[0]
            self.assertTrue(math.isfinite(item['ranking']['final_score']))
            self.assertEqual(item['ranking']['cosine_similarity'], -1)

    def test_pool_is_expanded_without_budget_and_bounded(self):
        self.assertEqual(candidate_pool_limit(1), 20)
        self.assertEqual(candidate_pool_limit(6), 30)
        self.assertEqual(candidate_pool_limit(100), 100)
        for invalid in (0, -1):
            with self.assertRaises(ValueError):
                candidate_pool_limit(invalid)
            with self.assertRaises(ValueError):
                select_ranked_products("q", [], None, limit=invalid)


class MaterialsTests(unittest.TestCase):
    def test_known_woods_get_provenance_and_scoped_properties(self):
        for material, alias in (("limba", "Лимба"), ("koto", "Koto"), ("hinoki", "Хиноки"), ("balsa", "Бальзы")):
            with self.subTest(material=material):
                context = material_context({"characteristics": {"Внешние слои": alias}})
                layer = context['layers'][0]
                self.assertEqual(layer['material'], material)
                self.assertEqual(layer['role'], 'outer')
                self.assertEqual(layer['catalog_text'], alias)
                self.assertTrue(layer['definition'])
                self.assertTrue(layer['sources'])
                self.assertIn("не измерения конкретной модели", context['scope'])

    def test_description_extracts_explicit_outer_and_core_without_marketing(self):
        product = {"description": "Непревзойдённый контроль. Состав: центральный слой Бальзы, внешние слои Лимба. Лучшая ракетка в мире!"}
        facts = material_facts(product)
        self.assertEqual({(f['material'], f['role']) for f in facts}, {('limba', 'outer'), ('balsa', 'core')})
        self.assertTrue(all(f['catalog_field'] == 'description' for f in facts))
        self.assertNotIn("Лучшая", json.dumps(material_context(product), ensure_ascii=False))

    def test_structured_layers_take_precedence_and_unspecified_is_not_outer(self):
        facts = material_facts({"characteristics": {"Внешние слои": "Koto", "Состав слоев": "Limba"}, "description": "Внешние слои Хиноки."})
        self.assertIn(('koto', 'outer'), {(f['material'], f['role']) for f in facts})
        self.assertIn(('limba', 'unspecified'), {(f['material'], f['role']) for f in facts})
        self.assertNotIn('hinoki', [f['material'] for f in facts])

    def test_inner_outer_require_explicit_structure_and_conflicts_are_withheld(self):
        for layout in ('Inner Carbon', 'Outer Carbon'):
            context = material_context({"characteristics": {"Конструкция": layout}})
            self.assertEqual(context['layers'][0]['material'], layout.split()[0].lower())
        context = material_context({"description": "Конструкция Inner Carbon со слоями у ядра."})
        self.assertEqual(context['layers'][0]['material'], 'inner')
        self.assertIsNone(material_context({"characteristics": {"Конструкция": "Inner или Outer"}}))
        self.assertIsNone(material_context({"characteristics": {"Конструкция": "Inner Carbon", "Расположение карбона": "Outer"}}))

    def test_no_guesses_from_name_reviews_layer_count_or_handle(self):
        for product in (
            {"name": "Hinoki Innerforce Carbon", "characteristics": {"Количество слоев": "5+2"}},
            {"description": "Butterfly Innerforce Layer ALC — легендарное основание."},
            {"characteristics": {"Конструкция": "С композитом"}},
            {"community_evidence": {"text": "Мягкая Лимба"}},
            {"characteristics": {"Материал ручки": "Hinoki"}},
            {"description": "Внешние слои не Limba."},
            {"description": "Внешние слои неизвестны, Limba не указана."},
            {"description": "Внешние слои возможно Limba."},
            {"description": "Конструкция предположительно Inner Carbon."},
            {"characteristics": {"Конструкция": "Без композита"}, "description": "Конструкция Inner Carbon."},
        ):
            with self.subTest(product=product):
                self.assertIsNone(material_context(product))

    def test_adjacent_material_roles_without_comma_do_not_bleed_into_each_other(self):
        facts = material_facts({"description": "Центральный слой Balsa и внешние слои Limba."})
        self.assertEqual({(f['material'], f['role']) for f in facts}, {('limba', 'outer'), ('balsa', 'core')})

    def test_russian_catalog_layer_placement_enriches_inner(self):
        context = material_context({'description': 'Слои карбона у основания Stiga Cybershape Carbon имеют внутреннее расположение, находясь около центрального деревянного слоя.'})
        self.assertEqual(context['layers'][0]['material'], 'inner')
        self.assertIn('Слои карбона', context['layers'][0]['catalog_text'])

    def test_prompt_uses_ontology_but_does_not_expose_ranking_or_raw_description(self):
        product = blade("DONIC Appelgren Allplay", description="Внешние слои Лимба. Абсолютный чемпион скорости!", ranking={"final_score": 999})
        catalog = json.loads(build_rag_prompt("Новичок", [product]).split('CATALOG_DATA:\n')[1])
        self.assertEqual(catalog[0]['material_reference']['layers'][0]['material'], 'limba')
        self.assertNotIn('ranking', catalog[0])
        self.assertNotIn('description', catalog[0])
        self.assertNotIn('Абсолютный чемпион', str(catalog))
        self.assertNotIn('similarity', catalog[0])

    def test_material_and_handle_queries_are_specific_enough_to_search(self):
        for criterion in ('Limba', 'Кото', 'Хиноки', 'Бальза', 'Dotec', 'пробковая ручка'):
            self.assertFalse(requires_clarification(f"Посоветуй основание {criterion}"))


class RetrievalIntegrationTests(unittest.TestCase):
    def test_eval_defaults_and_fingerprints_protect_historical_checkpoints(self):
        self.assertEqual(DEFAULT_OUTPUT.name, 'enrichment_ab_results_v7.json')
        self.assertEqual(DEFAULT_PROTOCOL.name, 'enrichment_ab_protocol_v7.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'fixture.json'
            fixture.write_text('{}')
            args = SimpleNamespace(repeats=3, limit=6, evidence_file=fixture, cases=fixture, protocol=fixture)
            config = _configuration(args, model='test', evidence_count=0, case_ids=['test'], generation_delay=15)
            checkpoint = root / 'historical.json'
            checkpoint.write_text(json.dumps({'schema_version': 2, 'configuration': config}))
            original = checkpoint.read_bytes()
            self.assertEqual(_load_checkpoint(checkpoint, config)['configuration'], config)
            changed = copy.deepcopy(config)
            changed['retrieval_code_sha256']['popularity.py'] = 'changed weights'
            with self.assertRaisesRegex(ValueError, 'конфигурация checkpoint отличается'):
                _load_checkpoint(checkpoint, changed)
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_service_and_eval_rank_same_full_pool_and_keep_evidence_on_product_id(self):
        candidates = [blade("DONIC Original Dotec ALL", .9, product_id=2)]
        candidates += [blade(f"Ordinary {i}", .85, product_id=10+i) for i in range(6)]
        candidates += [blade("DONIC Appelgren Allplay", .72, product_id=79)]
        evidence = {79: {"reviews_sampled": 5}}
        question = "Я новичок, хочу контрольное основание"
        with contextlib.ExitStack() as stack:
            for name in ('require_setting', 'generation_model', 'GeminiEmbeddingClient', 'GeminiGenerationClient'):
                stack.enter_context(patch('vsitemssrag.rag.service.' + name))
            stack.enter_context(patch('vsitemssrag.rag.service.load_community_evidence', return_value=evidence))
            database = stack.enter_context(patch('vsitemssrag.rag.service.Database.from_env'))
            db = database.return_value.__enter__.return_value
            db.match_products.return_value = candidates
            advisor = stack.enter_context(patch('vsitemssrag.rag.service.generate_advisor_answer', return_value='Allplay [1].'))
            result = ask_catalog(question)
            self.assertEqual(db.match_products.call_args.kwargs['limit'], 30)
            self.assertTrue(db.match_products.call_args.kwargs['only_available'])
            self.assertEqual(result.cited_products[0][1]['product_id'], 79)
            self.assertEqual(advisor.call_args.args[2][0]['community_evidence'], evidence[79])
        with patch('evals.run_enrichment_ab.GeminiEmbeddingClient'), patch('evals.run_enrichment_ab.Database.from_env') as database, contextlib.redirect_stdout(io.StringIO()):
            db = database.return_value.__enter__.return_value
            db.match_products.return_value = candidates
            rows = _retrieve_cases([{'id': 'smoke', 'question': question}], api_key='test', timeout=1, limit=6, evidence_by_id=evidence)
            self.assertEqual(db.match_products.call_args.kwargs['limit'], 30)
        self.assertEqual([p['name'] for p in rows[0]['products']], [p['name'] for p in result.products])
        self.assertEqual(rows[0]['enriched_product_ids'], [79])
