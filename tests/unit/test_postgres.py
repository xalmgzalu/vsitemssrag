import unittest

from vsitemssrag.storage.postgres import product_content_hash, variant_source_key


class ProductContentHashTests(unittest.TestCase):
    def test_hash_is_stable_for_characteristic_order(self):
        first = {
            "name": "Blade",
            "brand": "Brand",
            "description": "Description",
            "price": 1000,
            "currency": "RUB",
            "characteristics": {"Скорость": "9", "Контроль": "7"},
        }
        second = {
            **first,
            "characteristics": {"Контроль": "7", "Скорость": "9"},
        }

        self.assertEqual(product_content_hash(first), product_content_hash(second))

    def test_hash_changes_when_content_changes(self):
        product = {"name": "Blade", "price": 1000}
        changed = {"name": "Blade", "price": 1100}

        self.assertNotEqual(
            product_content_hash(product),
            product_content_hash(changed),
        )


class VariantSourceKeyTests(unittest.TestCase):
    def test_prefers_sku(self):
        self.assertEqual(
            variant_source_key({"sku": "ABC-123", "name": "Blade FL"}),
            "sku:ABC-123",
        )

    def test_falls_back_to_stable_name_hash(self):
        first = variant_source_key({"name": "Blade FL"})
        second = variant_source_key({"name": "Blade FL"})

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("name:"))


if __name__ == "__main__":
    unittest.main()
