import unittest

from homesocial.compositional_report import (
    CompositionalReport,
    render_compositional_report,
)


class CompositionalReportTests(unittest.TestCase):
    def test_render_contains_all_report_fields(self):
        rendered = render_compositional_report(
            CompositionalReport(
                need=0,
                severity=1,
                trend=2,
                confidence=2,
                cause=0,
            )
        )

        self.assertIn("need=need_food", rendered)
        self.assertIn("severity=low", rendered)
        self.assertIn("trend=worsening", rendered)
        self.assertIn("confidence=high", rendered)
        self.assertIn("cause=body_event", rendered)


if __name__ == "__main__":
    unittest.main()
