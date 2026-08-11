import unittest

from vsitemssrag.cli.main import build_parser


class CliParserTests(unittest.TestCase):
    def test_scrape_command(self):
        args = build_parser().parse_args(
            ["scrape", "--storage", "postgres", "--limit", "3"]
        )

        self.assertEqual(args.command, "scrape")
        self.assertEqual(args.storage, "postgres")
        self.assertEqual(args.limit, 3)

    def test_database_command(self):
        args = build_parser().parse_args(["db", "status"])

        self.assertEqual(args.command, "db")
        self.assertEqual(args.database_command, "status")

    def test_embed_command_defaults_to_safe_limit(self):
        args = build_parser().parse_args(["embed"])

        self.assertEqual(args.command, "embed")
        self.assertEqual(args.limit, 3)
        self.assertFalse(args.force)

    def test_search_command(self):
        args = build_parser().parse_args(
            ["search", "контрольное основание", "--limit", "7"]
        )

        self.assertEqual(args.command, "search")
        self.assertEqual(args.query, "контрольное основание")
        self.assertEqual(args.limit, 7)

    def test_ask_command(self):
        args = build_parser().parse_args(
            ["ask", "что выбрать новичку?", "--limit", "8"]
        )

        self.assertEqual(args.command, "ask")
        self.assertEqual(args.question, "что выбрать новичку?")
        self.assertEqual(args.limit, 8)
        self.assertEqual(args.max_output_tokens, 1200)


if __name__ == "__main__":
    unittest.main()
