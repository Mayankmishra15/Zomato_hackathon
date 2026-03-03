"""
Project offline model performance → business KPIs.
Uses elasticity assumptions grounded in food delivery benchmarks.
"""


def project_business_impact(
    precision_at_k,
    baseline_acceptance_rate=0.12,
    avg_item_price=120
):
    """
    Parameters
    ----------
    precision_at_k : float   model Precision@8
    baseline_acceptance_rate : float  current acceptance rate (12% assumed)
    avg_item_price : float   average add-on item price in INR

    Returns dict with projected metrics.
    """
    model_acceptance_rate = precision_at_k  # P@K ≈ expected acceptance rate
    lift_ratio = model_acceptance_rate / baseline_acceptance_rate

    aov_lift_pct = (model_acceptance_rate - baseline_acceptance_rate) * avg_item_price
    c2o_improvement = min(0.03, (model_acceptance_rate - baseline_acceptance_rate) * 0.15)
    attach_rate_lift = model_acceptance_rate - baseline_acceptance_rate

    return {
        "baseline_acceptance_rate": f"{baseline_acceptance_rate*100:.1f}%",
        "model_acceptance_rate": f"{model_acceptance_rate*100:.1f}%",
        "relative_lift": f"{(lift_ratio-1)*100:.1f}%",
        "projected_aov_lift_inr": f"INR {aov_lift_pct:.1f}",
        "c2o_improvement": f"+{c2o_improvement*100:.2f}%",
        "attach_rate_lift": f"+{attach_rate_lift*100:.1f}pp",
    }
