# LiteLLM integration

`examples/litellm-aliases.yaml` provides six aliases for an OpenAI-compatible DeepSeek endpoint. Replace `BACKEND_HOST` with a name reachable from the LiteLLM process, and review the configuration against the LiteLLM version you deploy. The example does not contain an API credential or a private address.

The aliases are `deepseek-v4.1-flash-1017`, `deepseek-v4.1-flash-1017-codex`, and each of those names with `-refusal` or `-norefusal` appended. The default aliases omit `cache_salt`, so the backend's global default applies. The refusal aliases send `refusal:2.5`; the no-refusal aliases send `refusal:0`. A client-provided `extra_body.cache_salt` can override an alias's configured body value if the LiteLLM version and route preserve that request field. Confirm this behavior with a request capture against your exact proxy release before relying on it.

The three Codex aliases set LiteLLM's `use_chat_completions_api` bridge for the Codex request format. All three non-Codex aliases fall back to `qwen3.8-flash-next` and then `gpt-6-luna`; all three Codex aliases fall back to `qwen3.8-flash-next-codex` and then `gpt-6-luna`. A fallback is another model and does not retain DeepSeek's lambda projection. Make the fallback policy and user-visible behavior explicit for your deployment.

## Example request override

Using an OpenAI-compatible client, send the per-request override in the provider-specific body:

```python
from openai import OpenAI

client = OpenAI(base_url="http://LITELLM_HOST:4000/v1", api_key="YOUR_LITELLM_KEY")
response = client.chat.completions.create(
    model="deepseek-v4.1-flash-1017-refusal",
    messages=[{"role": "user", "content": "Explain the topic briefly."}],
    extra_body={"cache_salt": "refusal:0"},
)
```

The example key is a placeholder. Keep LiteLLM credentials in the deployment's normal secret store. The request override selects a projection value; it is not an authorization control.

## Verify the route

Before enabling the aliases for users, confirm from the LiteLLM-to-backend request that the selected model, Responses/chat route, metadata, fallback chain, and `cache_salt` are what you expect. Run distinct `0` and `2.5` requests concurrently to verify isolation. A LiteLLM-side successful response alone does not prove the backend received the intended value.
