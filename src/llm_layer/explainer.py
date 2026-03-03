"""
LLM-powered recommendation explainer.
Uses GPT-4o-mini (or local model) to generate natural language reasons
for why an item is being recommended.

Falls back to rule-based template if LLM unavailable.
"""
import os

try:
    from src.config import ITEM_CATEGORIES, MEAL_SLOTS
except ImportError:
    from config import ITEM_CATEGORIES, MEAL_SLOTS


def get_rule_based_explanation(row):
    """Deterministic fallback explanation."""
    cat = ITEM_CATEGORIES.get(int(row.get("item_category", 3)), "item")
    meal = MEAL_SLOTS.get(int(row.get("meal_slot", 3)), "your meal")
    reasons = []
    if row.get("missing_drink"):
        reasons.append("complete your meal with a refreshing drink")
    if row.get("missing_dessert"):
        reasons.append("add a sweet finish")
    if row.get("missing_side"):
        reasons.append("pair well with your main course")
    if row.get("cuisine_match"):
        reasons.append("match your cuisine preference")
    if row.get("is_bestseller") or row.get("candidate_is_bestseller"):
        reasons.append("a customer favourite at this restaurant")
    if not reasons:
        reasons.append(f"popular choice for {meal}")
    return "Recommended because: " + ", ".join(reasons[:2]) + "."


def get_llm_explanation(row, client=None):
    """
    Optional: call OpenAI/Anthropic API for a richer explanation.
    If API key not set, falls back to rule-based.
    """
    if client is None or not os.getenv("OPENAI_API_KEY"):
        return get_rule_based_explanation(row)
    prompt = f"""
You are a food recommendation assistant for a delivery app.
Explain in ONE friendly sentence (max 20 words) why this add-on is a great choice:
- Cart already has: main={row.get('has_main_in_cart')}, side={row.get('has_side_in_cart')}, drink={row.get('has_drink_in_cart')}
- Recommended item: category={ITEM_CATEGORIES.get(int(row.get('item_category', 3)))}, veg={row.get('candidate_is_veg')}, bestseller={row.get('candidate_is_bestseller')}
- Context: {MEAL_SLOTS.get(int(row.get('meal_slot', 3)))} time
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return get_rule_based_explanation(row)
