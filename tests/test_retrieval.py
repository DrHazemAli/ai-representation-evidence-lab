from representation_evidence.retrieval import assemble_context, demo_candidates
from representation_evidence.policy.evaluator import PromotionPolicy, assemble_context_with_policy, candidate_from_mapping


def test_context_promotion_accepts_only_current_authorized_policy():
    context, evidence = assemble_context(demo_candidates(), tenant="A", max_tokens=32)

    assert "manual review" in context
    assert "auto-approved" not in context

    decisions = {
        item["id"]: item["decision"]
        for item in evidence["payload"]["trace"]
    }

    assert decisions["policy_current"] == "accepted"
    assert decisions["poisoned_near_neighbor"] == "rejected_wrong_authority"
    assert decisions["other_tenant_policy"] == "rejected_wrong_tenant"
    assert decisions["deleted_policy"] == "rejected_deleted"


def test_context_promotion_policy_matches_demo_boundary():
    policy = PromotionPolicy.from_mapping(
        {
            "name": "test",
            "version": 1,
            "required": {
                "tenant_match": True,
                "trusted_source": True,
                "fresh": True,
                "deleted": False,
                "intended_use": "answer_grounding",
                "allowed_authorities": ["approved_policy"],
            },
            "max_tokens": 32,
        }
    )

    context, evidence = assemble_context_with_policy(demo_candidates(), tenant="A", policy=policy)

    assert "manual review" in context
    assert evidence["payload"]["policy"]["name"] == "test"
    assert evidence["payload"]["accepted_count"] == 1


def test_candidate_loader_does_not_treat_false_string_as_true():
    candidate = candidate_from_mapping(
        {
            "id": "candidate",
            "score": 1.0,
            "text": "text",
            "source_authority": "approved_policy",
            "tenant": "A",
            "trusted": "false",
            "fresh": "true",
            "deleted": "false",
            "tokens": 1,
        }
    )

    assert candidate.trusted is False
    assert candidate.deleted is False
