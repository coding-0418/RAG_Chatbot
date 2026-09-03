from rag import contains_pii, is_investment_advice_query


class TestInvestmentAdviceGuardrail:
    def test_blocks_direct_advice_request(self):
        assert is_investment_advice_query("Should I invest in SBI Small Cap Fund?")

    def test_blocks_comparison_request(self):
        assert is_investment_advice_query("Which fund is better, Bluechip or Contra?")

    def test_blocks_return_prediction(self):
        assert is_investment_advice_query("Can you predict returns for the next year?")

    def test_blocks_recommendation_request(self):
        assert is_investment_advice_query("Please recommend a good mutual fund for me.")

    def test_allows_factual_question(self):
        assert not is_investment_advice_query("What is the expense ratio of SBI Bluechip Fund?")

    def test_allows_process_question(self):
        assert not is_investment_advice_query("How do I download my capital gains statement?")


class TestPrivacyGuardrail:
    def test_blocks_pan_number(self):
        assert contains_pii("My PAN is ABCDE1234F, can you check my account?")

    def test_blocks_email(self):
        assert contains_pii("Send the report to investor@example.com")

    def test_blocks_phone_number(self):
        assert contains_pii("Call me at 9876543210 about my SIP")

    def test_blocks_account_number(self):
        assert contains_pii("My account number is 123456789012")

    def test_allows_factual_question(self):
        assert not contains_pii("What is the minimum SIP amount for SBI Contra Fund?")

    def test_allows_scheme_names_with_numbers(self):
        assert not contains_pii("What is the lock-in period of the ELSS scheme?")
