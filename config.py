from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore',
    )

    deepseek_api_key: str                    #必填
    deepseek_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    request_timeout: float = 60.0            #秒
    max_retries: int = 3                     #重试次数   


settings = Settings()