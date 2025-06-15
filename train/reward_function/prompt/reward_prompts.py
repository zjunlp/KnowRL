LLM_EVAL_PROMPT = """Your job is to grade a predicted answer based on a question and a gold target. Assign one grade: ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Key Principles:
- CORRECT: Fully contains essential information from the gold target without contradictions. Minor variations obstáculos (spelling, order, hedging) are okay if the core meaning is right.
- INCORRECT: Contains factual contradictions to the gold target, or is significantly incomplete. Hedged incorrect statements are still INCORRECT.
- NOT_ATTEMPTED: Explicitly states inability to answer (e.g., "I don't know"), asks for more context evasively, or gives a very vague non-answer.

Examples:

1. CORRECT:
   Question: Barack Obama's children?
   Gold: Malia Obama and Sasha Obama
   Predicted: sasha and malia obama (Correct - order/case don't matter)
   Predicted: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly called Malia and Sasha. (Correct - extra info okay if not contradictory)

2. INCORRECT:
   Question: Barack Obama's children?
   Gold: Malia and Sasha
   Predicted: Malia. (Incorrect - incomplete)
   Predicted: Malia, Sasha, and Susan. (Incorrect - adds wrong info)
   Predicted: I think it's Malia and Jackie. (Incorrect - factual contradiction despite hedging)

3. NOT_ATTEMPTED:
   Question: Barack Obama's children?
   Gold: Malia and Sasha
   Predicted: I don't know. (Not Attempted)
   Predicted: I need more context. (Not Attempted)

Important Considerations (Summary):
- Numbers: Must be correct to the gold target's significant figures (e.g., Gold "120k", Pred "115k" is CORRECT; "100k" is INCORRECT; "around 100k" is NOT_ATTEMPTED).
- Partial Gold Info: If gold has more info than the question asks, the prediction only needs to answer the question (e.g., Q: "Episode name?", Gold: "S7, E20: White Wedding", Pred: "White Wedding" is CORRECT).
- Inferred Info: Don't penalize omission of info clearly inferred from the question (e.g., Q: "OpenAI HQ city?", Gold: "San Francisco, California", Pred: "San Francisco" is CORRECT).
- Name Typos: Minor typos in names are okay if clearly the same person.

Grade the new example below. Respond with only "A" for CORRECT, "B" for INCORRECT, or "C" for NOT_ATTEMPTED. No extra text.

---
Question: {question}
Gold target: {gold_answer}
Predicted answer: {answer}
---
"""





