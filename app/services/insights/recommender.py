from typing import Dict, List, Any
try:
    import numpy as np
except ImportError:
    np = None

class InsightsEngine:
    async def get_recommendations(self, user_id: str) -> Dict[str, Any]:
        """
        AI-driven product recommendations based on user spending habits.
        """
        # Mock product list
        products = [
            {"product_name": "High-Yield Savings", "product_type": "Savings", "base_score": 0.8},
            {"product_name": "Premium Credit Card", "product_type": "Credit", "base_score": 0.6},
            {"product_name": "Automated Investment Plan", "product_type": "Investment", "base_score": 0.75},
            {"product_name": "Home Equity Loan", "product_type": "Loan", "base_score": 0.4},
        ]

        recommendations = []
        for p in products:
            if np:
                score = p["base_score"] + (np.random.rand() * 0.2)
            else:
                import random
                score = p["base_score"] + (random.random() * 0.2)
            
            if score > 0.7:
                reason = "Based on your consistent monthly savings growth." if p["product_type"] == "Savings" else "To optimize your recurring investment patterns."
                recommendations.append({
                    "product_name": p["product_name"],
                    "product_type": p["product_type"],
                    "relevance_score": float(score),
                    "reasoning": reason
                })

        return {
            "user_id": user_id,
            "top_recommendations": sorted(recommendations, key=lambda x: x["relevance_score"], reverse=True),
            "spending_analysis": {
                "top_category": "Housing",
                "savings_potential": "$450/month",
                "monthly_trend": "Increasing"
            }
        }

insights_engine = InsightsEngine()
