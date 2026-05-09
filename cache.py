"""
utils/scheduler.py
Background job scheduler using APScheduler.
Runs fixture fetching, model updates, result settlement automatically.
"""
import os
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


class JobScheduler:
    """Manages all background jobs."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._setup_jobs()

    def _setup_jobs(self):
        # ── Every 30 min: refresh fixture cache ─────────────────
        self.scheduler.add_job(
            self._refresh_fixtures,
            IntervalTrigger(minutes=30),
            id="refresh_fixtures",
            name="Refresh Fixtures",
            replace_existing=True,
        )

        # ── Every 3 hours: settle finished predictions ───────────
        self.scheduler.add_job(
            self._settle_predictions,
            IntervalTrigger(hours=3),
            id="settle_predictions",
            name="Settle Predictions",
            replace_existing=True,
        )

        # ── Daily at 03:00 UTC: rebuild dataset ──────────────────
        self.scheduler.add_job(
            self._rebuild_dataset,
            CronTrigger(hour=3, minute=0),
            id="rebuild_dataset",
            name="Rebuild Training Dataset",
            replace_existing=True,
        )

        # ── Weekly Sunday 04:00 UTC: retrain ML models ───────────
        self.scheduler.add_job(
            self._retrain_models,
            CronTrigger(day_of_week="sun", hour=4, minute=0),
            id="retrain_models",
            name="Retrain ML Models",
            replace_existing=True,
        )

        print("[Scheduler] All jobs configured")

    async def _refresh_fixtures(self):
        """Refresh today's fixture cache."""
        try:
            from services.data_service.fetcher import DataService
            from datetime import date
            ds = DataService()
            today = date.today().isoformat()
            fixtures = await ds.get_fixtures(today)
            print(f"[Scheduler] Refreshed fixtures: {len(fixtures)} matches")
        except Exception as e:
            print(f"[Scheduler] Fixture refresh error: {e}")

    async def _settle_predictions(self):
        """Settle finished predictions."""
        try:
            from services.result_settlement import ResultSettlement
            from datetime import date, timedelta
            settler = ResultSettlement()
            yesterday = (date.today() - timedelta(days=1)).isoformat()
            result = settler.settle_predictions(yesterday)
            print(f"[Scheduler] Settlement: {result}")
        except Exception as e:
            print(f"[Scheduler] Settlement error: {e}")

    async def _rebuild_dataset(self):
        """Rebuild training dataset with recent matches."""
        try:
            from services.dataset_builder import DatasetBuilder
            from datetime import date
            builder = DatasetBuilder()
            season = date.today().year
            # Only fetch last 2 weeks of new data
            count = builder.build_dataset(league_ids=[39, 140, 135, 78, 61], seasons=[season])
            print(f"[Scheduler] Dataset rebuilt: {count} new fixtures")
        except Exception as e:
            print(f"[Scheduler] Dataset rebuild error: {e}")

    async def _retrain_models(self):
        """Retrain ML models with latest data."""
        try:
            from services.dataset_builder import DatasetBuilder
            from services.feature_engineering import FeatureEngineer
            from services.model_trainer import ModelTrainer

            builder  = DatasetBuilder()
            df_raw   = builder.get_dataframe()

            if len(df_raw) < 200:
                print(f"[Scheduler] Not enough data to retrain: {len(df_raw)} rows")
                return

            engineer = FeatureEngineer()
            df_feat  = engineer.transform(df_raw)
            trainer  = ModelTrainer()
            results  = trainer.train_all(df_feat, engineer)
            print(f"[Scheduler] Models retrained: {list(results.keys())}")

            # Reload predictor with new models
            from services.ml_predictor import get_predictor
            import services.ml_predictor as pred_module
            pred_module._predictor = None  # Force reload
            get_predictor()

        except Exception as e:
            print(f"[Scheduler] Retrain error: {e}")

    def start(self):
        self.scheduler.start()
        print(f"[Scheduler] Started at {datetime.utcnow().isoformat()} UTC")

    def stop(self):
        self.scheduler.shutdown()
        print("[Scheduler] Stopped")


# Singleton
_scheduler: JobScheduler = None

def get_scheduler() -> JobScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = JobScheduler()
    return _scheduler
