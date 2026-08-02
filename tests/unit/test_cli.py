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


if __name__ == "__main__":
    unittest.main()
