"""
Centralized MLflow tracking for RAG experiments.

Tracks: chunking params, embedding models, retrieval metrics, eval scores.
All tracking is wrapped in try/except so MLflow failures never crash the pipeline.

MLflow concepts:
  Experiment -> Named group of related runs (e.g., "F16_RAG_Pipeline")
  Run        -> Single execution with params, metrics, artifacts
  Params     -> Hyperparameters (logged once per run, strings)
  Metrics    -> Numeric values (logged per step for time-series tracking)
  Artifacts  -> Files (chunks.json, eval_results.json, parsed output)

Local usage (no server needed):
  mlflow ui --port 5000
  Open: http://localhost:5000

LATER -> remote tracking URI for team collaboration (e.g., http://mlflow-server:5000)
LATER -> model registry for versioning embedder/reranker/generator configs
"""
import mlflow
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from . import config
from .utils import logger


class MLflowTracker:
    """
    Singleton MLflow tracker for RAG pipeline experiments.

    Design decision: singleton pattern ensures all pipeline stages (parse,
    chunk, embed, index, eval) log to the same active run without passing
    the tracker as a parameter through every function call.

    Usage:
        tracker = MLflowTracker(experiment_name="F16_RAG_v2")
        with tracker.start_run(run_name="hierarchical_chunks"):
            tracker.log_params({"chunk_size": 1000, "overlap": 100})
            tracker.log_metrics({"context_precision": 0.85})
            tracker.log_artifact("output/chunks.json")
    """

    def __init__(self, experiment_name: str = "F16_RAG_Pipeline"):
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow with tracking URI and experiment."""
        try:
            tracking_uri = config.MLFLOW_TRACKING_URI or "./mlruns"
            mlflow.set_tracking_uri(tracking_uri)

            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=config.MLFLOW_ARTIFACT_LOCATION or None
                )

            mlflow.set_experiment(self.experiment_name)

            logger.info("mlflow_initialized",
                        experiment=self.experiment_name,
                        tracking_uri=tracking_uri)
        except Exception as e:
            logger.error("mlflow_setup_failed", error=str(e))
            raise

    def start_run(self, run_name: Optional[str] = None, tags: Dict[str, str] = None):
        """
        Start MLflow run with auto-generated name if not provided.

        Args:
            run_name: Optional custom run name (default: timestamp)
            tags: Optional tags for categorization (e.g., {"stage": "indexing"})

        Returns:
            MLflow active run context manager (use with `with` statement)
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        default_tags = {
            "pipeline_version": config.PIPELINE_VERSION or "1.0",
            "env": config.ENV or "dev",
        }
        if tags:
            default_tags.update(tags)

        logger.info("mlflow_run_start", run_name=run_name, tags=default_tags)
        return mlflow.start_run(run_name=run_name, tags=default_tags)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters (hyperparameters, config values). Strings only in MLflow."""
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            logger.info("mlflow_params_logged", count=len(params))
        except Exception as e:
            logger.error("mlflow_param_logging_failed", error=str(e))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics with optional step for time-series tracking.

        Args:
            metrics: Dict of metric_name -> numeric value
            step: Optional step number (for tracking metric evolution across epochs/runs)

        Non-numeric values are skipped with a warning (MLflow requires float/int).
        """
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
                else:
                    logger.warning("skipped_non_numeric_metric", key=key, value=value)
            logger.info("mlflow_metrics_logged", count=len(metrics))
        except Exception as e:
            logger.error("mlflow_metric_logging_failed", error=str(e))

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log file or directory as MLflow artifact.

        Args:
            local_path: Path to file or directory
            artifact_path: Optional subdirectory within the artifact store
        """
        try:
            path = Path(local_path)
            if not path.exists():
                logger.warning("artifact_not_found", path=str(path))
                return

            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path)
            else:
                mlflow.log_artifacts(str(path), artifact_path)

            logger.info("mlflow_artifact_logged", path=str(path))
        except Exception as e:
            logger.error("mlflow_artifact_logging_failed", path=local_path, error=str(e))

    def log_model_info(self, model_name: str, model_version: str, params: Optional[Dict] = None):
        """Log model metadata as params (e.g., embedder_version, reranker_model)."""
        model_info = {f"{model_name}_version": model_version}
        if params:
            for k, v in params.items():
                model_info[f"{model_name}_{k}"] = v
        self.log_params(model_info)

    def log_pipeline_run(self, pdf_path: str, chunk_count: int, timing: Dict[str, float]):
        """
        Convenience method: log all standard RAG pipeline params in one call.

        Logs config params + timing metrics for a complete indexing run.
        Call this after a successful run_pipeline.py execution.

        Args:
            pdf_path: Path to the indexed PDF
            chunk_count: Total chunks produced
            timing: Dict with keys: parse_time, chunk_time, embed_time, index_time, total_time
        """
        self.log_params({
            "pdf_path": str(pdf_path),
            "embed_model": config.EMBED_MODEL,
            "gen_model": config.GEN_MODEL,
            "reranker_model": config.RERANKER_MODEL,
            "max_chunk_tokens": config.MAX_CHUNK_TOKENS,
            "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            "embedding_dim": config.EMBEDDING_DIM,
            "top_k_initial": config.TOP_K_INITIAL,
            "top_k_final": config.TOP_K_FINAL,
        })
        self.log_metrics({
            "total_chunks": float(chunk_count),
            **{k: float(v) for k, v in timing.items() if isinstance(v, (int, float))},
        })

    def log_eval_results(self, ragas_scores: Dict[str, float], retrieval_scores: Dict[str, float]):
        """
        Log RAGAS evaluation scores as MLflow metrics.

        Prefixes ragas_ and retrieval_ to avoid name collisions.
        Useful for comparing eval runs across different chunking/model configs.

        Args:
            ragas_scores: Dict from ragas.evaluate() (context_precision, faithfulness, etc.)
            retrieval_scores: Dict from calculate_retrieval_metrics() (vector/rerank score stats)
        """
        ragas_prefixed = {f"ragas_{k}": v for k, v in ragas_scores.items()
                          if isinstance(v, (int, float))}
        retrieval_prefixed = {f"retrieval_{k}": v for k, v in retrieval_scores.items()
                              if isinstance(v, (int, float))}
        self.log_metrics({**ragas_prefixed, **retrieval_prefixed})

    def get_run_url(self) -> Optional[str]:
        """
        Return the MLflow UI URL for the current active run.
        Useful for displaying a clickable link in the Streamlit UI.

        Returns:
            URL string like "http://localhost:5000/#/experiments/1/runs/abc123"
            or None if no active run.
        """
        try:
            run = mlflow.active_run()
            if run is None:
                return None
            tracking_uri = config.MLFLOW_TRACKING_URI or "./mlruns"
            # Only construct URL for HTTP tracking URIs (not local file paths)
            if tracking_uri.startswith("http"):
                experiment_id = run.info.experiment_id
                run_id = run.info.run_id
                return f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
            return None
        except Exception:
            return None

    def end_run(self, status: str = "FINISHED"):
        """
        End current run with status.

        Args:
            status: One of ['FINISHED', 'FAILED', 'KILLED']
        """
        try:
            mlflow.end_run(status=status)
            logger.info("mlflow_run_ended", status=status)
        except Exception as e:
            logger.error("mlflow_end_run_failed", error=str(e))


# Singleton instance (module-level)
_tracker = None


def get_tracker(experiment_name: str = "F16_RAG_Pipeline") -> MLflowTracker:
    """
    Get or create the singleton MLflow tracker.

    Singleton pattern: ensures all pipeline stages share the same
    experiment context without re-initializing MLflow on each call.
    Thread-safe for single-process use (Streamlit runs single-threaded).
    LATER -> use threading.Lock for multi-threaded FastAPI serving.
    """
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker(experiment_name)
    return _tracker