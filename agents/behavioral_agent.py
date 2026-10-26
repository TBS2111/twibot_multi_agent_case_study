from preprocessing import features as F


class BehavioralAgent:
    """Shared-URL, interest and social-interaction viewpoints (three sub-agents in one)."""
    name = "behavioral"

    def run(self, state):
        users, cfg = state["users"], state["cfg"]
        url = F.sets_to_csr(F.url_sets(users))
        interest = F.sets_to_csr(F.interest_sets(users, cfg))
        social = F.sets_to_csr(F.social_sets(users))
        links = F.link_edges(users)
        feat = F.behavior_stats(users)
        return {"url_mat": url, "interest_mat": interest, "social_mat": social,
                "link_edges": links, "beh_feat": feat,
                "log": [f"[behavioral] url_vocab={url.shape[1]} interest_vocab={interest.shape[1]} "
                        f"social_vocab={social.shape[1]} links={len(links)}"]}
