"""
News sentiment pipeline for the Multi-Dimensional Return Forecasting system.

This module provides the ``SentimentDataFetcher`` class which fetches financial
news headlines (via *GNews* or synthetic generation) and scores them using
FinBERT (``ProsusAI/finbert``).  When FinBERT or its dependencies are
unavailable, a lightweight rule-based / random fallback is used so the
downstream pipeline always receives complete sentiment data.

Example:
    >>> from src.data.sentiment_data import SentimentDataFetcher
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> fetcher = SentimentDataFetcher(config)
    >>> sentiment = fetcher.fetch_all_sentiment()
"""

import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Company-specific headline templates for synthetic data
# ---------------------------------------------------------------------------
_POSITIVE_TEMPLATES: List[str] = [
    "{company} reports strong Q{q} results, beating analyst estimates",
    "{company} shares surge on robust revenue growth",
    "{company} announces record quarterly profit",
    "Analysts upgrade {company} citing strong fundamentals",
    "{company} expands operations with new strategic partnership",
    "{company} raises dividend payout for FY{year}",
    "{company} wins major contract worth ₹{amount} crore",
    "Foreign institutional investors increase stake in {company}",
    "{company} reports better-than-expected earnings guidance",
    "{company} launches innovative product line, market reacts positively",
]

_NEGATIVE_TEMPLATES: List[str] = [
    "{company} faces headwinds amid rising input costs",
    "Analysts downgrade {company} on margin pressure concerns",
    "{company} reports disappointing Q{q} numbers",
    "{company} shares fall after weak quarterly guidance",
    "Regulatory concerns weigh on {company} stock",
    "{company} faces supply chain disruptions impacting production",
    "{company} under pressure as competition intensifies",
    "Market sell-off drags {company} to 52-week low",
    "{company} profit drops amid challenging macro environment",
    "Investor concerns grow over {company} debt levels",
]

_NEUTRAL_TEMPLATES: List[str] = [
    "{company} announces board meeting for Q{q} results",
    "{company} to consider dividend in upcoming board meeting",
    "{company} holds annual general meeting",
    "{company} completes routine organisational restructuring",
    "{company} files quarterly regulatory report with SEBI",
    "{company} participates in industry conference",
    "Market watch: {company} stock trades flat on low volume",
    "{company} management discusses outlook at investor day",
    "{company} appoints new independent director to board",
    "{company} schedules analyst call for earnings review",
]

# Sentiment volatility profile per ticker (higher = more extreme headlines)
_SENTIMENT_VOLATILITY: Dict[str, float] = {
    "RELIANCE.NS": 0.55,
    "HDFCBANK.NS": 0.45,
    "INFY.NS": 0.50,
    "TATAMOTORS.NS": 0.70,  # Cyclical – more volatile sentiment
    "BHARTIARTL.NS": 0.50,
    "HINDUNILVR.NS": 0.40,  # Defensive – more stable sentiment
}


class SentimentDataFetcher:
    """Fetches financial news and generates sentiment scores using FinBERT.

    The pipeline is:  headlines → FinBERT scoring → daily aggregation.
    Both headline fetching and sentiment scoring have graceful fallbacks
    for environments where GNews or PyTorch / Transformers are not available.

    Attributes:
        config: Parsed ``config.yaml`` dictionary.
        tickers: List of Yahoo Finance ticker symbols.
        ticker_names: Mapping of ticker symbol → company name.
        start_date: Start of the data window.
        end_date: End of the data window.
        raw_data_path: Directory for persisting parquet files.
        model_name: HuggingFace model identifier for FinBERT.
        max_headlines_per_day: Cap on headlines ingested per day.
        batch_size: Batch size for transformer inference.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the sentiment pipeline from the master configuration.

        The FinBERT model is **not** loaded at construction time — it is
        loaded lazily on the first call to :py:meth:`compute_sentiment_scores`.

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        self.config = config

        self.tickers: list[str] = config.get("stocks", {}).get("tickers", [])
        self.ticker_names: dict[str, str] = config.get("stocks", {}).get(
            "names", {}
        )

        dates_cfg = config.get("dates", {})
        self.start_date: str = dates_cfg.get("start", "2020-01-01")
        self.end_date: str = dates_cfg.get("end", "2025-12-31")

        paths_cfg = config.get("paths", {})
        self.raw_data_path: str = paths_cfg.get("raw_data", "data/raw")
        self.raw_dir: str = paths_cfg.get("raw_sentiment", str(Path(self.raw_data_path) / "sentiment"))
        self.processed_path: str = paths_cfg.get("processed_data", "data/processed")

        sentiment_cfg = config.get("sentiment", {})
        self.model_name: str = sentiment_cfg.get(
            "model_name", "ProsusAI/finbert"
        )
        self.max_headlines_per_day: int = sentiment_cfg.get(
            "max_headlines_per_day", 10
        )
        self.batch_size: int = sentiment_cfg.get("batch_size", 16)
        # GNews.io API key: from config or GNEWS_API_KEY env var (env takes precedence)
        self.gnews_api_key: str = (
            os.environ.get("GNEWS_API_KEY") or sentiment_cfg.get("gnews_api_key", "")
        ).strip()
        # Finnhub API key: from config or FINNHUB_API_KEY env var (env takes precedence)
        self.finnhub_api_key: str = (
            os.environ.get("FINNHUB_API_KEY") or sentiment_cfg.get("finnhub_api_key", "")
        ).strip()
        # News source priority
        self.news_source_priority: list[str] = sentiment_cfg.get(
            "news_source_priority", ["finnhub", "gnews_io", "gnews", "synthetic"]
        )

        # Lazy-loaded FinBERT pipeline
        self._finbert_pipeline: Optional[Any] = None
        self._finbert_available: Optional[bool] = None
        
        # Lazy-loaded Finnhub client
        self._finnhub_client: Optional[Any] = None
        self._finnhub_available: Optional[bool] = None

        logger.info(
            "SentimentDataFetcher initialised – %d tickers, model=%s",
            len(self.tickers),
            self.model_name,
        )

    # ------------------------------------------------------------------
    # FinBERT model management
    # ------------------------------------------------------------------

    def _load_finbert(self) -> None:
        """Load the FinBERT model via HuggingFace ``transformers.pipeline``.

        The model is cached after the first successful load.  If
        ``transformers`` or ``torch`` are not installed the flag
        ``_finbert_available`` is set to ``False`` and all subsequent
        scoring calls will use the fallback.
        """
        if self._finbert_available is not None:
            return  # already attempted

        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading FinBERT model: %s …", self.model_name)
            self._finbert_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )
            self._finbert_available = True
            logger.info("FinBERT model loaded successfully")

        except ImportError as exc:
            self._finbert_available = False
            logger.warning(
                "transformers / torch not installed – FinBERT unavailable "
                "(%s). Using fallback scoring.",
                exc,
            )
        except Exception as exc:
            self._finbert_available = False
            logger.warning(
                "Failed to load FinBERT (%s). Using fallback scoring.", exc
            )

    # ------------------------------------------------------------------
    # News headline fetching
    # ------------------------------------------------------------------

    def fetch_news_headlines(
        self,
        company_name: str,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch news headlines for a company.

        Attempts sources in priority order: Finnhub → GNews.io → GNews → Synthetic.
        Falls back gracefully if any source fails or is unavailable.

        Args:
            company_name: Full company name for search queries.
            ticker: Yahoo Finance ticker symbol.
            start_date: Start date ``'YYYY-MM-DD'``.
            end_date: End date ``'YYYY-MM-DD'``.

        Returns:
            ``pd.DataFrame`` with columns ``['Date', 'Headline', 'Source']``.
        """
        # Try each source in priority order
        for source in self.news_source_priority:
            if source == "finnhub" and self.finnhub_api_key:
                df = self._try_finnhub(ticker, start_date, end_date)
                if df is not None and not df.empty:
                    logger.info(
                        "Finnhub returned %d headlines for %s", len(df), ticker
                    )
                    return df
            
            elif source == "gnews_io" and self.gnews_api_key:
                df = self._try_gnews_io(company_name, start_date, end_date)
                if df is not None and not df.empty:
                    logger.info(
                        "GNews.io API returned %d headlines for %s", len(df), ticker
                    )
                    return df
            
            elif source == "gnews":
                df = self._try_gnews(company_name, start_date, end_date)
                if df is not None and not df.empty:
                    logger.info(
                        "GNews returned %d headlines for %s", len(df), ticker
                    )
                    return df

        logger.info(
            "Falling back to synthetic headlines for %s", ticker
        )
        return self._generate_synthetic_headlines(ticker, start_date, end_date)

    def _load_finnhub(self) -> None:
        """Load the Finnhub client lazily.

        The client is cached after the first successful load. If the
        finnhub library is not installed, _finnhub_available is set to False.
        """
        if self._finnhub_available is not None:
            return  # already attempted

        try:
            import finnhub

            logger.info("Initializing Finnhub client…")
            self._finnhub_client = finnhub.Client(api_key=self.finnhub_api_key)
            self._finnhub_available = True
            logger.info("Finnhub client initialized successfully")

        except ImportError as exc:
            self._finnhub_available = False
            logger.debug(
                "finnhub-python not installed – Finnhub unavailable (%s)", exc
            )
        except Exception as exc:
            self._finnhub_available = False
            logger.debug("Failed to initialize Finnhub client (%s). Moving to next source.", exc)

    def _try_finnhub(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Attempt to fetch headlines via the Finnhub API.

        Args:
            ticker: Stock ticker symbol (e.g., 'INFY' or 'RELIANCE').
            start_date: Start date ``'YYYY-MM-DD'``.
            end_date: End date ``'YYYY-MM-DD'``.

        Returns:
            ``pd.DataFrame`` with headlines, or ``None`` on failure.
        """
        if not self.finnhub_api_key:
            return None

        try:
            self._load_finnhub()

            if not self._finnhub_available or self._finnhub_client is None:
                return None

            # Fetch company news from Finnhub
            articles = self._finnhub_client.company_news(
                ticker, _from=start_date, to=end_date
            )

            if not articles:
                return None

            records = []
            for art in articles:
                # Parse datetime
                pub_timestamp = art.get("datetime")
                if pub_timestamp:
                    pub_date = pd.to_datetime(pub_timestamp, unit="s", errors="coerce")
                else:
                    pub_date = pd.to_datetime(art.get("datetime"), errors="coerce")

                if pd.isna(pub_date):
                    continue

                headline = art.get("headline", "")
                if not headline or not headline.strip():
                    continue

                source = art.get("source", "Finnhub")

                records.append(
                    {
                        "Date": pub_date.normalize(),
                        "Headline": headline,
                        "Source": source,
                    }
                )

            if not records:
                return None

            return pd.DataFrame(records)

        except Exception as exc:
            logger.debug("Finnhub API fetch failed for %s: %s", ticker, exc)
            return None


    def _try_gnews_io(
        self,
        company_name: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch headlines via GNews.io REST API (requires API key)."""
        if not self.gnews_api_key:
            return None
        try:
            import urllib.parse
            import urllib.request

            query = f"{company_name} stock"
            from_ts = f"{start_date}T00:00:00.000Z"
            to_ts = f"{end_date}T23:59:59.999Z"
            params = {
                "q": query,
                "lang": "en",
                "country": "in",
                "max": min(100, max(10, self.max_headlines_per_day * 30)),
                "from": from_ts,
                "to": to_ts,
                "apikey": self.gnews_api_key,
            }
            url = "https://gnews.io/api/v4/search?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Pipeline/1.0)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = __import__("json").loads(resp.read().decode())
            articles = data.get("articles") or []
            if not articles:
                return None
            records = []
            for art in articles:
                pub_at = art.get("publishedAt") or ""
                pub_date = pd.to_datetime(pub_at, errors="coerce")
                if pd.isna(pub_date):
                    continue
                src = art.get("source") or {}
                source_name = src.get("name", "Unknown") if isinstance(src, dict) else str(src)
                records.append(
                    {
                        "Date": pub_date.normalize(),
                        "Headline": art.get("title", ""),
                        "Source": source_name,
                    }
                )
            if not records:
                return None
            return pd.DataFrame(records)
        except Exception as exc:
            logger.debug("GNews.io API fetch failed: %s", exc)
            return None

    def _try_gnews(
        self,
        company_name: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Attempt to fetch headlines via the *GNews* library.

        Args:
            company_name: Search query string.
            start_date: Start date.
            end_date: End date.

        Returns:
            ``pd.DataFrame`` with headlines, or ``None`` on failure.
        """
        try:
            from gnews import GNews

            gn = GNews(
                language="en",
                country="IN",
                start_date=tuple(int(x) for x in start_date.split("-")),
                end_date=tuple(int(x) for x in end_date.split("-")),
                max_results=100,
            )

            articles = gn.get_news(f"{company_name} stock")
            if not articles:
                return None

            records = []
            for art in articles:
                pub_date = pd.to_datetime(
                    art.get("published date", ""), errors="coerce"
                )
                if pd.isna(pub_date):
                    continue
                # gnews returns 'publisher' as either a dict (with 'title') or a string
                pub = art.get("publisher") or ""
                source = (
                    pub.get("title", "Unknown") if isinstance(pub, dict) else str(pub).strip() or "Unknown"
                )
                records.append(
                    {
                        "Date": pub_date.normalize(),
                        "Headline": art.get("title", ""),
                        "Source": source,
                    }
                )

            if not records:
                return None

            return pd.DataFrame(records)

        except ImportError:
            logger.debug("GNews library not installed")
            return None
        except Exception as exc:
            logger.debug("GNews fetch failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Synthetic headline generation
    # ------------------------------------------------------------------

    def _generate_synthetic_headlines(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Generate synthetic but realistic news headlines.

        Creates approximately 2-5 headlines per business day with a mix
        of positive, negative, and neutral sentiments.  The sentiment
        distribution varies per stock (e.g. ``TATAMOTORS.NS`` is more
        volatile).

        Args:
            ticker: Yahoo Finance symbol.
            start_date: Start date ``'YYYY-MM-DD'``.
            end_date: End date ``'YYYY-MM-DD'``.

        Returns:
            ``pd.DataFrame`` with columns ``['Date', 'Headline', 'Source']``.
        """
        rng = random.Random(hash(ticker))
        np_rng = np.random.RandomState(abs(hash(ticker)) % (2**31))
        company = self.ticker_names.get(ticker, ticker.replace(".NS", ""))

        bdays = pd.bdate_range(start=start_date, end=end_date)
        vol_factor = _SENTIMENT_VOLATILITY.get(ticker, 0.5)

        records: List[Dict[str, object]] = []
        sources = [
            "Economic Times",
            "Moneycontrol",
            "LiveMint",
            "Business Standard",
            "NDTV Profit",
            "Reuters India",
            "Bloomberg Quint",
        ]

        for day in bdays:
            # 2-5 headlines per day
            n_headlines = rng.randint(2, 5)

            for _ in range(n_headlines):
                # Sentiment class probabilities (adjusted by vol_factor)
                r = np_rng.random()
                # More volatile stocks have wider positive/negative split
                pos_threshold = 0.35 + vol_factor * 0.1
                neg_threshold = pos_threshold + 0.25 + vol_factor * 0.1

                if r < pos_threshold:
                    template = rng.choice(_POSITIVE_TEMPLATES)
                elif r < neg_threshold:
                    template = rng.choice(_NEGATIVE_TEMPLATES)
                else:
                    template = rng.choice(_NEUTRAL_TEMPLATES)

                headline = template.format(
                    company=company,
                    q=((day.month - 1) // 3) + 1,
                    year=day.year,
                    amount=rng.randint(500, 5000),
                )

                records.append(
                    {
                        "Date": day.normalize(),
                        "Headline": headline,
                        "Source": rng.choice(sources),
                    }
                )

        df = pd.DataFrame(records)
        logger.info(
            "Generated %d synthetic headlines for %s (%s → %s)",
            len(df),
            ticker,
            start_date,
            end_date,
        )
        return df

    # ------------------------------------------------------------------
    # Sentiment scoring
    # ------------------------------------------------------------------

    def compute_sentiment_scores(
        self, headlines_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute daily sentiment scores from headlines.

        If FinBERT is available each headline is passed through the model;
        otherwise a lightweight fallback with slight positive bias is used.

        For each calendar day:
        * All headline-level scores are averaged.
        * The aggregate score is ``positive − negative`` in ``[-1, 1]``.

        Args:
            headlines_df: ``pd.DataFrame`` with columns
                ``['Date', 'Headline', 'Source']``.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with columns:
            ``Sentiment_Score``, ``Sentiment_Positive``,
            ``Sentiment_Negative``, ``Sentiment_Neutral``,
            ``Headline_Count``.
        """
        if headlines_df.empty:
            return pd.DataFrame(
                columns=[
                    "Sentiment_Score",
                    "Sentiment_Positive",
                    "Sentiment_Negative",
                    "Sentiment_Neutral",
                    "Headline_Count",
                ]
            )

        # Ensure FinBERT load has been attempted
        self._load_finbert()

        if self._finbert_available and self._finbert_pipeline is not None:
            scored = self._score_with_finbert(headlines_df)
        else:
            scored = self._score_fallback(headlines_df)

        # Aggregate to daily level
        daily = (
            scored.groupby("Date")
            .agg(
                Sentiment_Score=("score", "mean"),
                Sentiment_Positive=("positive", "mean"),
                Sentiment_Negative=("negative", "mean"),
                Sentiment_Neutral=("neutral", "mean"),
                Headline_Count=("score", "count"),
            )
            .sort_index()
        )

        return daily

    def _score_with_finbert(
        self, headlines_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Score headlines using the loaded FinBERT model.

        Processes headlines in batches and maps FinBERT labels to
        positive / negative / neutral probabilities.

        Args:
            headlines_df: Headlines DataFrame.

        Returns:
            DataFrame with per-headline scores.
        """
        headlines = headlines_df["Headline"].tolist()
        dates = headlines_df["Date"].tolist()

        all_scores: List[Dict[str, Any]] = []
        n_batches = (len(headlines) + self.batch_size - 1) // self.batch_size

        for i in tqdm(
            range(0, len(headlines), self.batch_size),
            total=n_batches,
            desc="FinBERT scoring",
            leave=False,
        ):
            batch = headlines[i : i + self.batch_size]
            batch_dates = dates[i : i + self.batch_size]

            try:
                results = self._finbert_pipeline(batch)  # type: ignore[misc]

                for j, res in enumerate(results):
                    label = res["label"].lower()
                    confidence = res["score"]

                    pos = confidence if label == "positive" else 0.0
                    neg = confidence if label == "negative" else 0.0
                    neu = confidence if label == "neutral" else 0.0

                    all_scores.append(
                        {
                            "Date": batch_dates[j],
                            "positive": pos,
                            "negative": neg,
                            "neutral": neu,
                            "score": pos - neg,  # net sentiment
                        }
                    )
            except Exception as exc:
                logger.warning("FinBERT batch failed: %s – using fallback", exc)
                for j in range(len(batch)):
                    fb = self._fallback_single_score()
                    fb["Date"] = batch_dates[j]
                    all_scores.append(fb)

        return pd.DataFrame(all_scores)

    def _score_fallback(
        self, headlines_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Lightweight fallback scoring when FinBERT is not available.

        Assigns random sentiment with a slight positive bias based on
        simple keyword heuristics.

        Args:
            headlines_df: Headlines DataFrame.

        Returns:
            DataFrame with per-headline scores.
        """
        logger.info(
            "Using fallback sentiment scoring for %d headlines",
            len(headlines_df),
        )
        rng = np.random.RandomState(42)
        records = []

        positive_keywords = {
            "strong", "surge", "record", "upgrade", "growth",
            "beat", "profit", "win", "raise", "innovative",
            "increase", "better", "positive", "expands",
        }
        negative_keywords = {
            "headwind", "downgrade", "disappoint", "fall", "weak",
            "pressure", "disruption", "sell-off", "drop", "concern",
            "low", "debt", "decline", "loss",
        }

        for _, row in headlines_df.iterrows():
            headline_lower = str(row["Headline"]).lower()
            words = set(headline_lower.split())

            pos_hits = len(words & positive_keywords)
            neg_hits = len(words & negative_keywords)

            if pos_hits > neg_hits:
                pos = 0.5 + rng.random() * 0.4
                neg = rng.random() * 0.15
                neu = 1.0 - pos - neg
            elif neg_hits > pos_hits:
                neg = 0.5 + rng.random() * 0.4
                pos = rng.random() * 0.15
                neu = 1.0 - pos - neg
            else:
                neu = 0.5 + rng.random() * 0.3
                pos = 0.1 + rng.random() * 0.2
                neg = 1.0 - pos - neu

            # Clamp
            pos = max(0.0, min(1.0, pos))
            neg = max(0.0, min(1.0, neg))
            neu = max(0.0, min(1.0, neu))
            total = pos + neg + neu
            if total > 0:
                pos /= total
                neg /= total
                neu /= total

            records.append(
                {
                    "Date": row["Date"],
                    "positive": pos,
                    "negative": neg,
                    "neutral": neu,
                    "score": pos - neg,
                }
            )

        return pd.DataFrame(records)

    @staticmethod
    def _fallback_single_score() -> Dict[str, float]:
        """Return a single random sentiment score dict (emergency fallback).

        Returns:
            Dictionary with keys ``positive``, ``negative``, ``neutral``,
            ``score``.
        """
        rng = np.random.RandomState()
        pos = rng.random() * 0.5 + 0.1
        neg = rng.random() * 0.3
        neu = 1.0 - pos - neg
        return {
            "positive": pos,
            "negative": neg,
            "neutral": max(0.0, neu),
            "score": pos - neg,
        }

    def load_cached(self) -> Dict[str, pd.DataFrame]:
        """Load previously fetched sentiment data from ``data/raw/sentiment/``.

        Reads parquet files matching the pattern
        ``{ticker}_sentiment.parquet`` for every ticker in the configured
        universe.

        Returns:
            A dictionary mapping each ticker to its cached
            ``pd.DataFrame``.  Tickers whose cache files are missing
            are silently skipped.
        """
        cached_data: Dict[str, pd.DataFrame] = {}
        sent_dir = Path(self.raw_dir)

        for ticker in self.tickers:
            # Try new path first, fall back to old path
            file_path = sent_dir / f"{ticker}_sentiment.parquet"
            if not file_path.exists():
                file_path = Path(self.raw_data_path) / f"{ticker}_sentiment.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    cached_data[ticker] = df
                    logger.info(
                        "Loaded cached %s sentiment (%d rows) from %s",
                        ticker,
                        len(df),
                        file_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load cached sentiment for %s: %s", ticker, exc
                    )
            else:
                logger.warning("No cached sentiment found for %s at %s", ticker, file_path)

        logger.info(
            "Loaded cached sentiment for %d / %d tickers",
            len(cached_data),
            len(self.tickers),
        )
        return cached_data

    # ------------------------------------------------------------------
    # Public batch API
    # ------------------------------------------------------------------

    def fetch_all_sentiment(self) -> Dict[str, pd.DataFrame]:
        """Fetch and score sentiment for every stock in the universe.

        For each ticker the method:
        1. Fetches (or generates) headlines.
        2. Scores them via FinBERT or fallback.
        3. Aggregates to daily granularity.
        4. Saves to ``data/raw/sentiment/{ticker}_sentiment.parquet``.

        Returns:
            Dictionary mapping ticker → daily sentiment ``pd.DataFrame``.
        """
        all_data: Dict[str, pd.DataFrame] = {}
        sent_dir = Path(self.raw_dir)
        sent_dir.mkdir(parents=True, exist_ok=True)

        for ticker in tqdm(self.tickers, desc="Processing sentiment"):
            try:
                company_name = self.ticker_names.get(
                    ticker, ticker.replace(".NS", "")
                )

                # Step 1: headlines
                headlines_df = self.fetch_news_headlines(
                    company_name=company_name,
                    ticker=ticker,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )

                # Step 2+3: score and aggregate
                sentiment_df = self.compute_sentiment_scores(headlines_df)

                if sentiment_df.empty:
                    logger.warning(
                        "No sentiment data produced for %s", ticker
                    )
                    continue

                all_data[ticker] = sentiment_df

                # Step 4: persist
                out_path = sent_dir / f"{ticker}_sentiment.parquet"
                sentiment_df.to_parquet(out_path, index=True)
                logger.info(
                    "Saved %s sentiment → %s (%d days)",
                    ticker,
                    out_path,
                    len(sentiment_df),
                )

            except Exception as exc:
                logger.error(
                    "Sentiment pipeline failed for %s: %s", ticker, exc
                )

        logger.info(
            "Sentiment processed for %d / %d tickers",
            len(all_data),
            len(self.tickers),
        )

        # Save processed (merged) version
        if all_data:
            proc_dir = Path(self.processed_path)
            proc_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for t, df in all_data.items():
                tmp = df.copy()
                tmp["ticker"] = t
                frames.append(tmp)
            merged = pd.concat(frames)
            proc_path = proc_dir / "sentiment_processed.parquet"
            merged.to_parquet(proc_path, index=True)
            logger.info("Saved processed sentiment data → %s", proc_path)

        return all_data
