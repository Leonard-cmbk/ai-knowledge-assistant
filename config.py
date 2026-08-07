import os
from dotenv import load_dotenv


load_dotenv()

# DeepSeek-V4-Flash API_KEY
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("缺少 DEEPSEEK_API_KEY， 请在 .env 中配置")

# DeepSeek-V4-Flash base_url
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")
# DeepSeek-V4-Flash model
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")