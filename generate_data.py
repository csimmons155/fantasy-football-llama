import json
import os

import openai

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# number of queries we want to generate with GPT
NUM_BATCHES = 100
dataset = []

print(f"Starting generation of {NUM_BATCHES * 10} trade scenarios...")

# Give GPT system level context
SYSTEM_PROMPT_PREV = """
You are a ruthlessly analytical Fantasy Football Expert (PPR format).
Your goal is to evaluate trades with mathematical precision and aggressive subjectivity.

**CRITICAL INSTRUCTION:**
You must correct for "Trade Directionality" logic.
- If "Team A sends Player X", then **Team B** has Player X.
- You must evaluate the trade based on **NET ASSETS RECEIVED**, not assets lost.

**Identity Rules:**
1. **No Hedging:** Never say "it depends on team needs." Pick a clear winner. If it's close, pick the side with the higher upside.
2. **Tone:** Arrogant but backed by data. Use short, punchy sentences.
3. **Statistical Reasoning:** You must reference specific metrics to justify your decision.
   - Use: "Target Share", "Air Yards", "Red Zone Efficiency", "Strength of Schedule (SOS)", "Regression to the Mean", "Positive/Negative Game Script".
   - Avoid: Generic terms like "he is a good player" or "he scores a lot."

**Output Structure:**
For every trade evaluation, you must simulate a "Value Calculation" before giving the final verdict.

"""

SYSTEM_PROMPT = """

You are a ruthlessly analytical Fantasy Football Expert (PPR format).
Your goal is to evaluate trades with mathematical precision and aggressive subjectivity.

**CRITICAL INSTRUCTION:**
You must correct for "Trade Directionality" logic.
- If "Team A sends Player X", then **Team B** has Player X.
- You must evaluate the trade based on **NET ASSETS RECEIVED**, not assets lost.

**Identity Rules:**
1. **No Hedging:** Never say "it depends on team needs." Pick a clear winner. If it's close, pick the side with the higher upside.
2. **Tone:** Arrogant but backed by data. Use short, punchy sentences.
3. **Statistical Reasoning:** You must reference specific metrics to justify your decision.
   - Use: "Target Share", "Air Yards", "Red Zone Efficiency", "Strength of Schedule (SOS)", "Regression to the Mean", "Positive/Negative Game Script".
   - Avoid: Generic terms like "he is a good player" or "he scores a lot."

**Output Structure:**
For every trade, you must strictly follow this Chain of Thought (CoT):
1. **The Ledger:** Explicitly list what each team *receives*.
   - Team A Gets: [List players/picks]
   - Team B Gets: [List players/picks]
2. **The Valuation:** Analyze the "Team Gets" column only.
   - Compare "Team A Gets" vs "Team B Gets".
3. **The Verdict:** Declare the winner based on who acquired the better assets.

"""

# Give GPT user level context that'll use for each batch

USER_PROMPT = """
Generate 10 distinct fantasy football trade scenarios involving active NFL players.
Focus on a 12-team PPR (Points Per Reception) league context.

Requirements:
1. Use real, active NFL players in the current season (make sure to also include niche players like rookies and back-up players in some instances too).
2. Include specific jargon: "ADP", "Target Share", "Regression to the mean", "Stacking", "Keeper value".
3. The 'output' must be an analytical breakdown of the trade, determining clearly who won. Be opinionated.
4. Include logic that explains which teams wins and which team loses, explicitly mentioning a statistical disparity.

Return ONLY a valid JSON array of objects. Do not use markdown formatting.
Each object must have exactly these fields:
- "input": A string describing the trade (e.g., "Team A sends Justin Jefferson; Team B sends...")
- "cot_ledger": A string explicitly stating: "Team A receives [Player list]. Team B receives [Player list]."
- "output": Your expert analysis and verdict, referencing the 'cot_ledger' to ensure the correct winner is picked.
"""

for i in range(NUM_BATCHES):
    print(f"Generating batch: {i + 1}/{NUM_BATCHES}...")

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.7,
        )

        # Clean up response
        content = completion.choices[0].message.content

        if not content:
            raise ValueError("API returned empty content")

        content = content.replace("```json", "").replace("```", "").strip()
        batch_data = json.loads(content)

        # iterates over every json object
        for item in batch_data:
            # an instruction is added so that the model learns this specific task (for Unsloth models)
            structured_row = {
                "instruction": "Evaluate this fantasy football trade.",
                "input": item["input"],
                "output": item["output"],
            }
            dataset.append(structured_row)

    except Exception as e:
        print(f"Error on batch {i}, {e}")

output_folder = "project-1/training_data"
output_filename = "fantasy_football_trades.jsonl"
full_path = os.path.join(output_folder, output_filename)

os.makedirs(output_folder, exist_ok=True)

with open(full_path, "w") as f:
    for e in dataset:
        json.dump(e, f)
        f.write("\n")

print(f"Succcessful completion; saved {len(dataset)} items to {output_filename}")
