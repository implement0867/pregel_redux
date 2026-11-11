from pydantic_settings import BaseSettings, SettingsConfigDict


class RunSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PREGEL_REDUX_", frozen=True)

    step_limit: int = 25
