import unittest

from vsitemssrag.enrichment.bladerubber import (
    aggregate_bladerubber_evidence,
    catalog_from_sitemap_urls,
    parse_product_page,
    parse_sitemap,
)
from vsitemssrag.enrichment.tabletennisdaily import (
    aggregate_review_evidence,
    classify_long_pips_relation,
    exact_model_match,
    match_score,
    parse_catalog_page,
    parse_reviews_page,
    select_pilot_products,
)


CATALOG_HTML = """
<div class="grid-item large-grid-item">
  <div class="grid-item-category-title-header"><a>Blades</a></div>
  <h3><a href="/equipment/acoustic-carbon-inner.1900/">Acoustic Carbon Inner</a></h3>
  <span class="ratingStars" title="4.67 star(s)"></span>
  <a class="u-muted" href="/equipment/acoustic-carbon-inner.1900/reviews">(3)</a>
</div>
<div class="grid-item large-grid-item">
  <h3><a href="/equipment/featured-review.1/">Featured review</a></h3>
</div>
"""


REVIEWS_HTML = """
<h1>Nittaku Acoustic Carbon Inner</h1>
<div class="block item-information--default">
  <dl class="pairs"><dt>Brand</dt><dd>Nittaku</dd></dl>
  <dl class="pairs"><dt>Reviews</dt><dd>2</dd></dl>
  <dl class="pairs"><dt>Rating</dt><dd><span class="ratingStars" title="4.50 star(s)"></span></dd></dl>
</div>
<div class="rms-review">
  <span class="ratingStars" title="5.00 star(s)"></span>
  <a href="/equipment/acoustic-carbon-inner.1900/review/10/"><time></time></a>
  <article class="message-body"><span class="review-title">Soft and controlled</span></article>
  <article class="message-body"><div class="bbWrapper">Flexible blade with long dwell time.</div></article>
  <div class="rms-review-fields">
    <dl><dt>Speed</dt><dd><div class="bar-value">8</div></dd></dl>
    <dl><dt>Control</dt><dd><div class="bar-value">9</div></dd></dl>
  </div>
</div>
<div class="rms-review">
  <span class="ratingStars" title="4.00 star(s)"></span>
  <a href="/equipment/acoustic-carbon-inner.1900/review/11/"><time></time></a>
  <article class="message-body"><div class="bbWrapper">Good control but still fast.</div></article>
  <div class="rms-review-fields">
    <dl><dt>Speed</dt><dd><div class="bar-value">6</div></dd></dl>
    <dl><dt>Control</dt><dd><div class="bar-value">8</div></dd></dl>
  </div>
</div>
"""


class TableTennisDailyParserTests(unittest.TestCase):
    def test_parses_only_product_cards(self):
        items = parse_catalog_page(CATALOG_HTML)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "Acoustic Carbon Inner")
        self.assertEqual(items[0].rating, 4.67)
        self.assertEqual(items[0].review_count, 3)

    def test_aggregates_structured_stats_without_raw_text(self):
        page = parse_reviews_page(
            REVIEWS_HTML,
            page_url="https://www.tabletennisdaily.com/equipment/acoustic-carbon-inner.1900/reviews",
        )
        evidence = aggregate_review_evidence(
            page,
            source_url="https://www.tabletennisdaily.com/equipment/acoustic-carbon-inner.1900/",
        )

        self.assertEqual(page.brand, "Nittaku")
        self.assertEqual(evidence["reviews_sampled"], 2)
        self.assertEqual(evidence["stats"]["speed"], {"mean": 7.0, "sample_size": 2})
        self.assertEqual(evidence["mentions"]["soft_feel"]["review_count"], 1)
        self.assertEqual(evidence["mentions"]["control"]["review_count"], 2)
        self.assertFalse(evidence["raw_review_text_retained"])
        self.assertNotIn("Soft and controlled", str(evidence))


BLADERUBBER_JSON_LD = """
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "Butterfly Diode Pro",
  "brand": {"@type": "Brand", "name": "Butterfly"},
  "aggregateRating": {"ratingValue": "9.0", "reviewCount": 2},
  "review": [
    {
      "@type": "Review",
      "reviewBody": "Soft defensive blade used with long pips OX.",
      "reviewRating": {"ratingValue": 10}
    },
    {
      "@type": "Review",
      "reviewBody": "Controlled chopping but enough speed for attack.",
      "reviewRating": {"ratingValue": 8}
    }
  ]
}
</script>
"""


class BladeRubberParserTests(unittest.TestCase):
    def test_parses_sitemap_product_urls(self):
        xml = """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://bladerubber.net/blades/butterfly/diode-pro</loc></url>
          <url><loc>https://bladerubber.net/blades/butterfly</loc></url>
          <url><loc>https://bladerubber.net/rubbers/test/model</loc></url>
        </urlset>"""

        urls = parse_sitemap(xml)
        items = catalog_from_sitemap_urls(urls)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "diode pro")

    def test_aggregates_json_ld_reviews_without_retaining_text(self):
        page = parse_product_page(
            BLADERUBBER_JSON_LD,
            page_url="https://bladerubber.net/blades/butterfly/diode-pro/reviews",
        )
        evidence = aggregate_bladerubber_evidence(
            page,
            source_url="https://bladerubber.net/blades/butterfly/diode-pro",
        )

        self.assertEqual(page.brand, "Butterfly")
        self.assertEqual(page.name, "Diode Pro")
        self.assertEqual(evidence["reviews_sampled"], 2)
        self.assertEqual(
            evidence["relations"]["long_pips"]["positive_review_count"],
            1,
        )
        self.assertEqual(evidence["mentions"]["defensive_play"]["review_count"], 2)
        self.assertNotIn("Soft defensive blade", str(evidence))

    def test_classifies_long_pips_direction_without_retaining_text(self):
        self.assertEqual(
            classify_long_pips_relation("Works very well with long pips OX."),
            "positive",
        )
        self.assertEqual(
            classify_long_pips_relation("This is not suitable for long pips."),
            "negative",
        )
        self.assertEqual(
            classify_long_pips_relation("Question about long pimples."),
            "unclear",
        )


class EnrichmentMatchingTests(unittest.TestCase):
    def test_matches_normalized_brand_and_model(self):
        score = match_score(
            "Основание NITTAKU Acoustic Carbon Inner",
            "Nittaku",
            "Acoustic Carbon Inner",
        )

        self.assertGreater(score, 0.95)

    def test_compound_words_match_but_model_variant_does_not(self):
        self.assertTrue(
            exact_model_match(
                "Основание DONIC Balsa Carbo Certran Fibre",
                "Donic",
                "BalsaCarbo Certran Fibre",
            )
        )
        self.assertFalse(
            exact_model_match(
                "Основание VICTAS Koji Matsushita OFF",
                "Victas",
                "Koji Matsushita",
            )
        )

    def test_selects_targets_then_deterministic_brand_sample(self):
        products = [
            {"id": 1, "name": "Target", "brand": "A", "source_url": "target"},
            {"id": 2, "name": "A1", "brand": "A", "source_url": "a1"},
            {"id": 3, "name": "A2", "brand": "A", "source_url": "a2"},
            {"id": 4, "name": "B1", "brand": "B", "source_url": "b1"},
        ]

        first = select_pilot_products(products, ["target"], limit=3)
        second = select_pilot_products(products, ["target"], limit=3)

        self.assertEqual(first, second)
        self.assertEqual(first[0]["source_url"], "target")
        self.assertEqual(first[0]["cohort"], "targeted")
        self.assertEqual({item["brand"] for item in first[1:]}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
