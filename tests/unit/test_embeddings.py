import unittest

from vsitemssrag.embeddings.documents import (
    build_product_document,
    build_search_query,
)
from vsitemssrag.embeddings.gemini import (
    GeminiEmbeddingError,
    parse_embedding_response,
)


class EmbeddingDocumentTests(unittest.TestCase):
    def test_builds_stable_semantic_document(self):
        document = build_product_document(
            {
                "name": "Основание Test",
                "brand": "TestBrand",
                "description": "Контрольное основание",
                "price": 5000,
                "available": True,
                "characteristics": {
                    "Бренд": "TestBrand",
                    "Скорость": "6",
                    "Контроль": "9",
                },
            }
        )

        self.assertTrue(document.startswith("title: Основание Test | text: "))
        self.assertEqual(document.count("Бренд: TestBrand"), 1)
        self.assertIn("Контроль: 9", document)
        self.assertIn("Описание: Контрольное основание", document)
        self.assertNotIn("5000", document)
        self.assertNotIn("available", document)

    def test_omits_description_that_only_repeats_name(self):
        document = build_product_document(
            {
                "name": "Основание Test",
                "description": "Основание Test",
                "characteristics": {},
            }
        )

        self.assertNotIn("Описание:", document)

    def test_formats_search_task(self):
        self.assertEqual(
            build_search_query("  контрольное   основание для новичка "),
            "task: search result | query: контрольное основание для новичка",
        )


class GeminiResponseTests(unittest.TestCase):
    def test_parses_singular_embedding_response(self):
        self.assertEqual(
            parse_embedding_response({"embedding": {"values": [0, 1]}}, 2),
            [0.0, 1.0],
        )

    def test_parses_plural_embedding_response(self):
        self.assertEqual(
            parse_embedding_response({"embeddings": [{"values": [1, 0]}]}, 2),
            [1.0, 0.0],
        )

    def test_rejects_wrong_dimensions(self):
        with self.assertRaises(GeminiEmbeddingError):
            parse_embedding_response({"embedding": {"values": [1]}}, 2)


if __name__ == "__main__":
    unittest.main()
