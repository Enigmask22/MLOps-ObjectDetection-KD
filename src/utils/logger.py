"""
Cấu hình hệ thống Logging chuẩn cho toàn bộ dự án MLOps.
==========================================================
Sử dụng thư viện logging thay vì print() để đảm bảo tính nhất quán,
khả năng truy vết và tuân thủ tiêu chuẩn observability.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Khởi tạo và cấu hình logger chuẩn cho module.

    Args:
        name: Tên định danh của logger (thường là __name__).
        log_file: Đường dẫn tệp log. Nếu None, chỉ ghi ra console.
        level: Mức độ log tối thiểu (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Định dạng chuỗi log.
        max_bytes: Kích thước tối đa của tệp log trước khi xoay vòng.
        backup_count: Số lượng tệp log dự phòng giữ lại.

    Returns:
        logging.Logger: Đối tượng logger đã được cấu hình.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tránh thêm handler trùng lặp khi gọi lại
    if logger.handlers:
        return logger

    formatter = logging.Formatter(log_format)

    # Handler ghi ra console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler ghi ra tệp (nếu được chỉ định)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Lấy logger đã được cấu hình theo tên module.

    Args:
        name: Tên module (thường dùng __name__).

    Returns:
        logging.Logger: Logger đã cấu hình sẵn.
    """
    return setup_logger(name, log_file="logs/app.log")
