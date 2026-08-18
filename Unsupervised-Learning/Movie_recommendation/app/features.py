import ast


def normalize_token(token: str) -> str:
    return token.replace(" ", "").lower()


def parse_names(json_like_str: str, key: str = "name") -> list[str]:
    try:
        items = ast.literal_eval(json_like_str)
    except (ValueError, SyntaxError):
        return []
    return [item[key] for item in items if key in item]


def extract_top_cast(cast_json_str: str, top_n: int) -> list[str]:
    try:
        cast = ast.literal_eval(cast_json_str)
    except (ValueError, SyntaxError):
        return []
    return [member["name"] for member in cast[:top_n] if "name" in member]


def extract_director(crew_json_str: str) -> str | None:
    try:
        crew = ast.literal_eval(crew_json_str)
    except (ValueError, SyntaxError):
        return None
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name")
    return None


def build_soup(
    genres: list[str],
    cast: list[str],
    director: str | None,
    keywords: list[str] | None = None,
    director_weight: int = 3,
) -> str:
    tokens = [normalize_token(g) for g in genres]
    tokens += [normalize_token(a) for a in cast]
    if director:
        tokens += [normalize_token(director)] * director_weight
    if keywords:
        tokens += [normalize_token(k) for k in keywords]
    return " ".join(tokens)
