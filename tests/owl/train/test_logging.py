from owl.train.logging import DebugLogger, WandbLogger


def test_wandb_logger_reports_exit_code() -> None:
    finished: list[int] = []

    class _Run:
        def finish(self, *, exit_code: int = 0) -> None:
            finished.append(exit_code)

    logger = WandbLogger.__new__(WandbLogger)
    logger._run = _Run()
    logger.close(exit_code=1)
    logger.close()

    assert finished == [1, 0]


def test_debug_logger_accepts_exit_code() -> None:
    logger = DebugLogger()

    logger.close(exit_code=1)
    logger.close()
