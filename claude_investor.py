import ast
import json
import os
from datetime import datetime, timedelta

import click
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

from weave import Model
import weave

from typing import Dict
from pydantic import Field

class ClaudeInvestor(Model):
    headers: Dict[str, str] = Field(default_factory=lambda: {
        "x-api-key": "",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })

    generate_ticker_ideas_prompt: str = "You are a financial analyst assistant. Generate a list of 5 ticker symbols for major companies in the {industry} industry, as a Python-parseable list."
    generate_ticker_ideas_user_message: str = "Please provide a list of 5 ticker symbols for major companies in the {industry} industry as a Python-parseable list. Only respond with the list, no other text."
    generate_ticker_ideas_model: str = "claude-3-haiku-20240307"

    get_sentiment_analysis_prompt: str = "You are a sentiment analysis assistant. Analyze the sentiment of the given news articles for {ticker} and provide a summary of the overall sentiment and any notable changes over time. Be measured and discerning. You are a skeptical investor."
    get_sentiment_analysis_user_message: str = "News articles for {ticker}:\n{news_text}\n\n----\n\nProvide a summary of the overall sentiment and any notable changes over time."
    get_sentiment_analysis_model: str = "claude-3-haiku-20240307"

    get_industry_analysis_prompt: str = "You are an industry analysis assistant. Provide an analysis of the {industry} industry and {sector} sector, including trends, growth prospects, regulatory changes, and competitive landscape. Be measured and discerning. Truly think about the positives and negatives of the stock. Be sure of your analysis. You are a skeptical investor."
    get_industry_analysis_user_message: str = "Provide an analysis of the {industry} industry and {sector} sector."
    get_industry_analysis_model: str = "claude-3-haiku-20240307"

    get_final_analysis_prompt: str = "You are a financial analyst providing a final investment recommendation for {ticker} based on the given data and analyses. Be measured and discerning. Truly think about the positives and negatives of the stock. Be sure of your analysis. You are a skeptical investor."
    get_final_analysis_user_message: str = "Ticker: {ticker}\n\nComparative Analysis:\n{comparisons}\n\nSentiment Analysis:\n{sentiment_analysis}\n\nAnalyst Ratings:\n{analyst_ratings}\n\nIndustry Analysis:\n{industry_analysis}\n\nBased on the provided data and analyses, please provide a comprehensive investment analysis and recommendation for {ticker}. Consider the company's financial strength, growth prospects, competitive position, and potential risks. Provide a clear and concise recommendation on whether to buy, hold, or sell the stock, along with supporting rationale."
    get_final_analysis_model: str = "claude-3-opus-20240229"

    rank_companies_prompt: str = "You are a financial analyst providing a ranking of companies in the {industry} industry based on their investment potential. Be discerning and sharp. Truly think about whether a stock is valuable or not. You are a skeptical investor."
    rank_companies_user_message: str = "Industry: {industry}\n\nCompany Analyses:\n{analysis_text}\n\nBased on the provided analyses, please rank the companies from most attractive to least attractive for investment. Provide a brief rationale for your ranking. In each rationale, include the current price (if available) and a price target."
    rank_companies_model: str = "claude-3-opus-20240229"

    def __init__(self, ANTHROPIC_API_KEY):
        super().__init__()
        self.headers["x-api-key"] = ANTHROPIC_API_KEY

    @weave.op()
    def generate_ticker_ideas(self, industry):
        messages = [
            {
                "role": "user",
                "content": self.generate_ticker_ideas_user_message.format(
                    industry=industry
                ),
            },
        ]

        data = {
            "model": self.generate_ticker_ideas_model,
            "max_tokens": 200,
            "temperature": 0.5,
            "system": self.generate_ticker_ideas_prompt.format(industry=industry),
            "messages": messages,
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages", headers=self.headers, json=data
        )
        response_text = response.json()["content"][0]["text"]

        ticker_list = ast.literal_eval(response_text)
        return [ticker.strip() for ticker in ticker_list]

    @weave.op()
    def get_stock_data(self, ticker, years):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=years * 365)

        stock = yf.Ticker(ticker)

        # Retrieve historical price data
        hist_data = stock.history(start=start_date, end=end_date)

        # Retrieve balance sheet
        balance_sheet = stock.balance_sheet

        # Retrieve financial statements
        financials = stock.financials

        # Retrieve news articles
        news = stock.news

        return hist_data, balance_sheet, financials, news

    @weave.op()
    def get_sentiment_analysis(self, ticker, news):
        news_text = ""
        for article in news:
            article_text = self.get_article_text(article["link"])
            timestamp = datetime.fromtimestamp(article["providerPublishTime"]).strftime(
                "%Y-%m-%d"
            )
            news_text += f"\n\n---\n\nDate: {timestamp}\nTitle: {article['title']}\nText: {article_text}"

        messages = [
            {
                "role": "user",
                "content": self.get_sentiment_analysis_user_message.format(
                    ticker=ticker, news_text=news_text
                ),
            },
        ]
        data = {
            "model": self.get_sentiment_analysis_model,
            "max_tokens": 2000,
            "temperature": 0.5,
            "system": self.get_sentiment_analysis_prompt.format(ticker=ticker),
            "messages": messages,
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages", headers=self.headers, json=data
        )
        response_text = response.json()["content"][0]["text"]

        return response_text

    @weave.op()
    def get_article_text(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            article_text = " ".join([p.get_text() for p in soup.find_all("p")])
            return article_text
        except Exception as e:
            print(f"Error retrieving article text: {e}")
            return "Error retrieving article text."

    @weave.op()
    def get_analyst_ratings(self, ticker):
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        if recommendations is None or recommendations.empty:
            return "No analyst ratings available."

        
        rating_summary = f"Analyst ratings for {ticker}:\n\n"
        
        for _, row in recommendations.iterrows():
            rating_summary += f"Period: {row['period']}\n"
            rating_summary += f"Strong Buy: {row['strongBuy']}\n"
            rating_summary += f"Buy: {row['buy']}\n"  
            rating_summary += f"Hold: {row['hold']}\n"
            rating_summary += f"Sell: {row['sell']}\n"
            rating_summary += f"Strong Sell: {row['strongSell']}\n\n"

        return rating_summary

    @weave.op()
    def get_industry_analysis(self, ticker):
        stock = yf.Ticker(ticker)
        industry = stock.info["industry"]
        sector = stock.info["sector"]

        messages = [
            {
                "role": "user",
                "content": self.get_industry_analysis_user_message.format(
                    industry=industry, sector=sector
                ),
            },
        ]
        data = {
            "model": self.get_industry_analysis_model,
            "max_tokens": 2000,
            "temperature": 0.5,
            "system": self.get_industry_analysis_prompt.format(
                industry=industry, sector=sector
            ),
            "messages": messages,
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages", headers=self.headers, json=data
        )
        response_text = response.json()["content"][0]["text"]

        return response_text

    @weave.op()
    def get_final_analysis(
        self,
        ticker,
        comparisons,
        sentiment_analysis,
        analyst_ratings,
        industry_analysis,
    ):

        messages = [
            {
                "role": "user",
                "content": self.get_final_analysis_user_message.format(
                    ticker=ticker,
                    comparisons=json.dumps(comparisons, indent=2),
                    sentiment_analysis=sentiment_analysis,
                    analyst_ratings=analyst_ratings,
                    industry_analysis=industry_analysis,
                ),
            },
        ]
        data = {
            "model": self.get_final_analysis_model,
            "max_tokens": 3000,
            "temperature": 0.5,
            "system": self.get_final_analysis_prompt.format(ticker=ticker),
            "messages": messages,
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages", headers=self.headers, json=data
        )
        response_text = response.json()["content"][0]["text"]

        return response_text

    @weave.op()
    def get_current_price(self, ticker):
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d", interval="1m")
        return data["Close"][-1]

    @weave.op()
    def rank_companies(self, industry, analyses, prices):

        analysis_text = "\n\n".join(
            f"Ticker: {ticker}\nCurrent Price: {prices.get(ticker, 'N/A')}\nAnalysis:\n{analysis}"
            for ticker, analysis in analyses.items()
        )

        messages = [
            {
                "role": "user",
                "content": self.rank_companies_user_message.format(
                    industry=industry, analysis_text=analysis_text
                ),
            },
        ]

        data = {
            "model": self.rank_companies_model,
            "max_tokens": 3000,
            "temperature": 0.5,
            "system": self.rank_companies_prompt.format(industry=industry),
            "messages": messages,
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages", headers=self.headers, json=data
        )
        response_text = response.json()["content"][0]["text"]

        return response_text

    @weave.op()
    def perform_ticker_analysis(self, ticker, years=1):
        print(f"\nAnalyzing {ticker}...")
        hist_data, balance_sheet, financials, news = self.get_stock_data(
            ticker, years
        )
        main_data = {
            "hist_data": hist_data,
            "balance_sheet": balance_sheet,
            "financials": financials,
            "news": news,
        }
        sentiment_analysis = self.get_sentiment_analysis(ticker, news)
        print(f"Sentiment Analysis for {ticker}: {sentiment_analysis}\n\n")
        analyst_ratings = self.get_analyst_ratings(ticker)
        print(f"Analyst Ratings for {ticker}: {analyst_ratings}\n\n")
        industry_analysis = self.get_industry_analysis(ticker)
        print(f"Industry Analysis for {ticker}: {industry_analysis}\n\n")
        final_analysis = self.get_final_analysis(
            ticker, {}, sentiment_analysis, analyst_ratings, industry_analysis
        )
        print(f"Final Analysis for {ticker}:\n{final_analysis}\n\n")
        current_price = self.get_current_price(ticker)
        print(f"Current Price for {ticker}: {current_price}\n\n")

        return final_analysis, current_price


    @weave.op()
    def predict(self, industry="artifical intelligence", years=1):
        # Generate ticker ideas for the industry
        tickers = self.generate_ticker_ideas(industry)
        print(f"\nTicker Ideas for {industry} Industry:")
        print(", ".join(tickers))

        # Perform analysis for each company
        analyses = {}
        prices = {}
        for ticker in tickers:
            try:
                analysis, current_price = self.perform_ticker_analysis(ticker, years)
                analyses[ticker] = analysis
                prices[ticker] = current_price
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                continue

        # Rank the companies based on their analyses
        ranking = self.rank_companies(industry, analyses, prices)
        return ranking, analyses, prices


@click.command()
@click.option("--industry", default="artificial intelligence", help="The industry to analyze.")
@click.option(
    "--years", default=1, help="The number of years of historical data to retrieve."
)
def main(industry, years):
    weave.init('claude-investor')
    investor = ClaudeInvestor(ANTHROPIC_API_KEY=os.environ["ANTHROPIC_API_KEY"])
    ranking, analyses, prices = investor.predict(industry=industry, years=years)
    print(f"Ranking for {industry} industry:")
    print(ranking)
    print("\nAnalyses:")
    print(json.dumps(analyses, indent=2))
    print("\nPrices:")
    print(json.dumps(prices, indent=2))

if __name__ == "__main__":
    main()