from app.generation.prompt import build_rag_prompt
from app.generation.hallucination import hallucination_score


def test_prompt_building():
    q = "What is employee leave policy?"
    ctx = ["Employees get 30 days leave annually"]

    prompt = build_rag_prompt(q, ctx)

    assert "Context" in prompt
    assert "User Question" in prompt
    assert "Answer" in prompt
    assert "Employees get 30 days" in prompt


def test_hallucination_score_reasonable():
    answer = "Employees receive 30 days paid leave."
    contexts = [
        "Employees get 30 days paid annual leave as per HR policy"
    ]

    score = hallucination_score(answer, contexts)

    assert score > 0.5
