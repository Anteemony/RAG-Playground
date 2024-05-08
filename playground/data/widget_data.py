

model_provider = {
                "mixtral-8x7b-instruct-v0.1": ["together-ai", "octoai", "replicate", "mistral-ai", "perplexity-ai",
                                               "anyscale", "fireworks-ai", "lepton-ai", "deepinfra", "aws-bedrock"],
                "llama-2-70b-chat": ["anyscale", "perplexity-ai", "together-ai", "replicate", "octoai", "fireworks-ai",
                                     "lepton-ai", "deepinfra", "aws-bedrock"],
                "llama-2-13b-chat": ["anyscale", "together-ai", "replicate", "octoai", "fireworks-ai", "lepton-ai",
                                     "deepinfra", "aws-bedrock"],
                "mistral-7b-instruct-v0.2": ["perplexity-ai", "together-ai", "mistral-ai", "replicate", "aws-bedrock",
                                             "octoai", "fireworks-ai"],
                "llama-2-7b-chat": ["anyscale", "together-ai", "replicate", "fireworks-ai", "lepton-ai", "deepinfra"],
                "codellama-34b-instruct": ["anyscale", "perplexity-ai", "together-ai", "octoai", "fireworks-ai",
                                           "deepinfra"],
                "gemma-7b-it": ["anyscale", "together-ai", "fireworks-ai", "lepton-ai", "deepinfra"],
                "mistral-7b-instruct-v0.1": ["anyscale", "together-ai", "fireworks-ai", "deepinfra"],
                "mixtral-8x22b-instruct-v0.1": ["mistral-ai", "together-ai", "fireworks-ai", "deepinfra"],
                "codellama-13b-instruct": ["together-ai", "octoai", "fireworks-ai"],
                "codellama-7b-instruct": ["together-ai", "octoai"], "yi-34b-chat": ["together-ai", "deepinfra"],
                "llama-3-8b-chat": ["together-ai", "fireworks-ai"], "llama-3-70b-chat": ["together-ai", "fireworks-ai"],
                "pplx-7b-chat": ["perplexity-ai"], "mistral-medium": ["mistral-ai"], "gpt-4": ["openai"],
                "pplx-70b-chat": ["perplexity-ai"], "gpt-3.5-turbo": ["openai"],
                "deepseek-coder-33b-instruct": ["together-ai"], "gemma-2b-it": ["together-ai"], "gpt-4-turbo": ["openai"],
                "mistral-small": ["mistral-ai"], "mistral-large": ["mistral-ai"], "claude-3-haiku": ["anthropic"],
                "claude-3-opus": ["anthropic"], "claude-3-sonnet": ["anthropic"]
}

dynamic_provider = ["lowest-input-cost", "lowest-output-cost", "lowest-itl", "lowest-ttft", "highest-tks-per-sec"]

model_reset_dict = {
                    "slider_model_temperature": "model_temperature"
                }

splitter_reset_dict = {
                    "slider_chunk_size": "chunk_size",
                    "slider_chunk_overlap": "chunk_overlap"
                }