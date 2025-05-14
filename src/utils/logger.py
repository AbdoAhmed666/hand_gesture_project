import logging
import os

# إنشاء مجلد logs لو مش موجود
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# اسم اللوج ثابت
log_filename = os.path.join(log_dir, "training.log")

# إعدادات اللوجر
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='a'),  # نسجل في نفس الملف
        logging.StreamHandler()  # يطبع في الكونسول كمان
    ]
)

# اللوجر الأساسي
logger = logging.getLogger("HGR")

# ✅ الدالة اللي بترجع اللوجر
def get_logger(name="HGR"):
    return logging.getLogger(name)
