"""Logging utilities for training and evaluation.

Provides structured logging setup and utilities for consistent
logging across the codebase.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# if TYPE_CHECKING:
#     from tqdm import tqdm


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.

    Configures logging to write to console and optionally to a file.

    Args:
        log_file: Optional path to log file
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string

    Example:
        >>> setup_logging('logs/training.log', level=logging.INFO)
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Training started")
    """
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing batch 10/100")
        >>> logger.warning("Metric not improving")
        >>> logger.error("Training failed")
    """
    return logging.getLogger(name)


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that works with tqdm progress bars.

    Prevents log messages from interfering with tqdm progress bars.

    Example:
        >>> from tqdm import tqdm
        >>>
        >>> # Setup logging with tqdm handler
        >>> logger = get_logger(__name__)
        >>> logger.addHandler(TqdmLoggingHandler())
        >>>
        >>> # Use with tqdm
        >>> for i in tqdm(range(100)):
        ...     logger.info(f"Processing {i}")  # Won't mess up progress bar
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record using tqdm.write."""
        try:
            import tqdm as tqdm_module

            msg = self.format(record)
            tqdm_module.tqdm.write(msg)
        except ImportError:
            # Fallback if tqdm not installed
            print(self.format(record))


class MetricLogger:
    """Simple logger for tracking metrics during training.

    Provides a cleaner interface for logging metrics with automatic
    formatting and aggregation.

    Example:
        >>> logger = MetricLogger()
        >>>
        >>> # Log metrics
        >>> logger.log('train/loss', 0.5, step=100)
        >>> logger.log('train/psnr', 25.3, step=100)
        >>> logger.log('val/psnr', 26.8, step=100)
        >>>
        >>> # Print summary
        >>> logger.print_summary()
        [Step 100] train/loss=0.5000, train/psnr=25.30, val/psnr=26.80
    """

    def __init__(self, delimiter: str = " | ") -> None:
        """Initialize metric logger.

        Args:
            delimiter: Delimiter between metrics when printing
        """
        self.delimiter = delimiter
        self.metrics: dict = {}
        self.logger = get_logger(self.__class__.__name__)

    def log(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value.

        Args:
            name: Metric name (e.g., 'train/loss', 'val/psnr')
            value: Metric value
            step: Optional step/epoch number
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append((step, value))

    def log_dict(
        self, metrics: dict, prefix: str = "", step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            prefix: Prefix to add to all metric names (e.g., 'train/')
            step: Optional step/epoch number

        Example:
            >>> logger.log_dict({'loss': 0.5, 'psnr': 25.3}, prefix='train/', step=100)
        """
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            self.log(full_name, value, step)

    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric.

        Args:
            name: Metric name

        Returns:
            Latest value or None if metric not found
        """
        if name not in self.metrics or not self.metrics[name]:
            return None
        return float(self.metrics[name][-1][1])

    def print_summary(self, step: Optional[int] = None) -> None:
        """Print summary of all metrics.

        Args:
            step: Optional step number to filter by
        """
        if not self.metrics:
            return

        parts = []

        for name, values in sorted(self.metrics.items()):
            if not values:
                continue

            # Get latest value (optionally filtered by step)
            if step is not None:
                filtered = [v for s, v in values if s == step]
                if not filtered:
                    continue
                value = filtered[-1]
            else:
                value = values[-1][1]

            parts.append(f"{name}={value:.4f}")

        if parts:
            summary = self.delimiter.join(parts)
            if step is not None:
                summary = f"[Step {step}] {summary}"
            self.logger.info(summary)

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics.clear()


__all__ = [
    "setup_logging",
    "get_logger",
    "TqdmLoggingHandler",
    "MetricLogger",
]
