from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Agent Portfolio Recommender"
    log_level: str = "INFO"

    # LLM
    llm_provider: str = "mock"  # mock | openai_compatible | ollama | qwen | kimi | n1n
    # 可选：给 secondary 模型单独指定 provider（用于双模型跨 provider 对比）
    secondary_llm_provider: str | None = None

    # OpenAI-compatible (OpenAI/DeepSeek/SiliconFlow/阿里千问兼容模式等)
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "deepseek-chat"
    secondary_model: str | None = None

    # Qwen(DashScope) convenience fields
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_api_key: str | None = None
    dashscope_model: str = "qwen-plus"

    # Kimi (Moonshot, OpenAI-compatible)
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    kimi_api_key: str | None = None
    kimi_model: str = "kimi-k2-thinking"

    # N1N aggregator (OpenAI-compatible gateway)
    n1n_base_url: str = "https://api.n1n.ai"
    n1n_api_key: str | None = None
    n1n_model: str = "gemini-3-pro"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    secondary_ollama_model: str | None = None

    # Data
    data_provider: str = "mock"  # mock | baostock | akshare | tushare
    tushare_token: str | None = None


settings = Settings()

