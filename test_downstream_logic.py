import unittest
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app_helpers import (
    extract_recommendation_template,
    extract_comparison_template,
    extract_statistical_template,
    extract_synthesis_template
)

class TestDownstreamLogic(unittest.TestCase):

    def test_extract_recommendation_template(self):
        print("\nTesting: test_extract_recommendation_template")
        mock_llm_response = """
        <RECOMMENDATIONS_START>
        <REC1>
        <ACTION>Consolidate vendors for office supplies.</ACTION>
        <JUSTIFICATION>High number of vendors with low spending per vendor.</JUSTIFICATION>
        <PRIORITY>HIGH</PRIORITY>
        </REC1>
        <REC2>
        <ACTION>Negotiate bulk discount with DELL INC.</ACTION>
        <JUSTIFICATION>High total spending ($1500.00) across 2 orders.</JUSTIFICATION>
        <PRIORITY>MEDIUM</PRIORITY>
        </REC2>
        </RECOMMENDATIONS_START>
        """
        expected_output = "Strategic Recommendations:\n\n1. Consolidate vendors for office supplies. (Priority: HIGH)\n   Justification: High number of vendors with low spending per vendor.\n\n2. Negotiate bulk discount with DELL INC. (Priority: MEDIUM)\n   Justification: High total spending ($1500.00) across 2 orders."
        actual_output = extract_recommendation_template(mock_llm_response)
        print(f"Expected:\n{expected_output}")
        print(f"Actual:\n{actual_output}")
        self.assertEqual(expected_output.strip(), actual_output.strip())
        print("Test Passed.")

    def test_extract_comparison_template(self):
        print("\nTesting: test_extract_comparison_template")
        mock_llm_response = """
        <COMPARISON_START>
        <SUMMARY>DELL INC has higher total spending, but IBM has a higher average order value.</SUMMARY>
        <VENDOR1>
        <NAME>DELL INC</NAME>
        <PERFORMANCE>Total spending: $1500.00, Order count: 2</PERFORMANCE>
        <STRENGTHS>Consistent order volume.</STRENGTHS>
        <CONCERNS>Lower average order value compared to IBM.</CONCERNS>
        </VENDOR1>
        <VENDOR2>
        <NAME>INTERNATIONAL BUSINESS MACHINES</NAME>
        <PERFORMANCE>Total spending: $2000.00, Order count: 1</PERFORMANCE>
        <STRENGTHS>High-value single orders.</STRENGTHS>
        <CONCERNS>Low order frequency.</CONCERNS>
        </VENDOR2>
        <RECOMMENDATION>Focus on DELL for regular purchases, and IBM for strategic high-value items.</RECOMMENDATION>
        </COMPARISON_START>
        """
        expected_output = "Summary: DELL INC has higher total spending, but IBM has a higher average order value.\n\n**DELL INC**\nPerformance: Total spending: $1500.00, Order count: 2\nStrengths: Consistent order volume.\nConcerns: Lower average order value compared to IBM.\n\n**INTERNATIONAL BUSINESS MACHINES**\nPerformance: Total spending: $2000.00, Order count: 1\nStrengths: High-value single orders.\nConcerns: Low order frequency.\n\nRecommendation: Focus on DELL for regular purchases, and IBM for strategic high-value items."
        actual_output = extract_comparison_template(mock_llm_response)
        print(f"Expected:\n{expected_output}")
        print(f"Actual:\n{actual_output}")
        self.assertEqual(expected_output.strip(), actual_output.strip())
        print("Test Passed.")

    def test_extract_statistical_template(self):
        print("\nTesting: test_extract_statistical_template")
        mock_llm_response = """
        <STATISTICAL_ANALYSIS>
        <SUMMARY>The overall spending shows a stable trend with a few high-value outliers.</SUMMARY>
        <FINDING1>The median order value is $195.00, which is significantly lower than the mean of $3044.01.</FINDING1>
        <FINDING2>The standard deviation is high, indicating large variance in order values.</FINDING2>
        <BUSINESS_IMPACT>The discrepancy between median and mean suggests that a few large purchases are skewing the average.</BUSINESS_IMPACT>
        <RECOMMENDATIONS>Investigate the high-value orders to understand the key drivers of cost.</RECOMMENDATIONS>
        </STATISTICAL_ANALYSIS>
        """
        expected_output = "Summary: The overall spending shows a stable trend with a few high-value outliers.\n\nKey Findings:\n1. The median order value is $195.00, which is significantly lower than the mean of $3044.01.\n2. The standard deviation is high, indicating large variance in order values.\n\nBusiness Impact: The discrepancy between median and mean suggests that a few large purchases are skewing the average.\n\nRecommendations: Investigate the high-value orders to understand the key drivers of cost."
        actual_output = extract_statistical_template(mock_llm_response)
        print(f"Expected:\n{expected_output}")
        print(f"Actual:\n{actual_output}")
        self.assertEqual(expected_output.strip(), actual_output.strip())
        print("Test Passed.")

    def test_extract_synthesis_template(self):
        print("\nTesting: test_extract_synthesis_template")
        mock_llm_response = """
        <RESPONSE_START>
        <ANSWER>The total spending for DELL INC was $1500.00 across 2 orders.</ANSWER>
        </RESPONSE_START>
        """
        expected_output = "The total spending for DELL INC was $1500.00 across 2 orders."
        actual_output = extract_synthesis_template(mock_llm_response)
        print(f"Expected:\n{expected_output}")
        print(f"Actual:\n{actual_output}")
        self.assertEqual(expected_output.strip(), actual_output.strip())
        print("Test Passed.")

if __name__ == '__main__':
    unittest.main()
