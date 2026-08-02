import unittest

from vsitemssrag.scraping.vistasport import parse_catalog_page, parse_product_page


CATALOG_HTML = """
<html>
  <head><link rel="next" href="?PAGEN_1=2"></head>
  <body>
    <div id="catalog-content">
      <div class="products-flex-item">
        <a class="name" href="/catalog/blades/test_blade.html">Тест</a>
        <a class="name" href="/catalog/blades/test_blade.html">Дубликат</a>
      </div>
    </div>
  </body>
</html>
"""


PRODUCT_HTML = """
<html>
  <body>
    <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "ProductGroup",
        "name": "Тестовое основание",
        "url": "https://www.vistasport.ru/catalog/blades/test_blade.html",
        "description": "Короткое описание...",
        "brand": {"@type": "Brand", "name": "TestBrand"},
        "image": ["https://www.vistasport.ru/image.jpg"],
        "hasVariant": [
          {
            "@type": "Product",
            "name": "Тестовое основание FL",
            "sku": "TEST-FL",
            "offers": {
              "@type": "Offer",
              "price": 1234,
              "priceCurrency": "RUB",
              "availability": "https://schema.org/InStock"
            }
          }
        ],
        "additionalProperty": [
          {"@type": "PropertyValue", "name": "Тип основания", "value": "OFF"}
        ]
      }
    </script>
    <div itemprop="description">
      Это полное описание тестового основания, взятое из HTML.
    </div>
  </body>
</html>
"""


class CatalogParserTests(unittest.TestCase):
    def test_extracts_unique_links_and_next_page(self):
        links, next_url = parse_catalog_page(
            CATALOG_HTML,
            "https://www.vistasport.ru/catalog/blades/",
        )

        self.assertEqual(
            links,
            ["https://www.vistasport.ru/catalog/blades/test_blade.html"],
        )
        self.assertEqual(
            next_url,
            "https://www.vistasport.ru/catalog/blades/?PAGEN_1=2",
        )


class ProductParserTests(unittest.TestCase):
    def test_extracts_product_group(self):
        product = parse_product_page(
            PRODUCT_HTML,
            "https://www.vistasport.ru/catalog/blades/test_blade.html",
        )

        self.assertEqual(product["name"], "Тестовое основание")
        self.assertEqual(product["price"], 1234)
        self.assertEqual(product["currency"], "RUB")
        self.assertTrue(product["available"])
        self.assertEqual(product["brand"], "TestBrand")
        self.assertEqual(product["characteristics"]["Тип основания"], "OFF")
        self.assertIn("полное описание", product["description"])
        self.assertEqual(product["variants"][0]["sku"], "TEST-FL")


if __name__ == "__main__":
    unittest.main()
