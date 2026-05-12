from agent import _sonnet_web_search

result = _sonnet_web_search(
    query="EQ-5D number of validated language translations EuroQol",
    instruction="Return JSON: {count: int, source_url: str}."
)
print(result)
