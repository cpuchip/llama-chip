# DiffusionGemma × pg-ai-stewards — detached trial

Model: `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (Q4_K_M, one RTX 4090). Model load: 8s. Runner: Unsloth's `llama-diffusion-gemma-visual-server`, driven over stdio (no Python).

## Summary

| Fixture | Stage | tok | DiffusionGemma tok/s | decode s | ref model |
|---|---|--:|--:|--:|---|
| 01-doc-plan-star-trek-guide | plan | 1133 | **145** | 7.8 | qwen3.6-35b-a3b |
| 02-content-star-trek-factions | build-content | 716 | **235** | 3.0 | qwen3.6-35b-a3b |
| 03-toolcall-weather | tool-call | 63 | **102** | 0.6 | (capability probe — no substrate reference) |

---

## 01-doc-plan-star-trek-guide — `plan`

**Task (binding question):**

> Create a 'New Player's Guide to Star Trek Adventures' PDF. Structure: 1. Introduction to the Game (Storytelling, Eras). 2. Core Mechanics (2d20, Attributes, Departments, Stress, Momentum/Threat). 3. Character Creation (Lifepath). 4. Starship Play. 5. Why Play? (Highlights vs D&D/STO). Use the 'star-trek' corpus for accurate terminology.

**DiffusionGemma** (1133 tok, 145 tok/s, 7.8s):

```
<|channel>thought
*   Topic: 'New Player's Guide to Star Trek Adventures' (RPG).
    *   Structure:
        1. Intro (Storytelling, Eras).
        2. Core Mechanics (2d20, Attributes, Departments, Stress, Momentum/Threat).
        3. Character Creation (Lifepath).
        4. Starship Play.
        5. Why Play? (vs D&D/STO).
    *   Constraint: Use'star-trek' corpus for terminology.
    *   Output: BUILD PLAN (not the document).

    *   *Output Format:* PDF.
    *   *Sections:*
        1.  *Introduction:* Narrative focus (collaborative storytelling), timeline (TOS, TNG, DS, Voyager, etc.).
        2.  *Core Mechanics:* 2d20 resolution system (Success/Failure vs. Difficulty), Attributes (Daring, Control, etc.), Departments (Command,Science, Ops), Stress pool, and economy (Momentum/Threat).
        3.  *Character Creation:* Lifepath system, Origins, Specializations, Talents.
        4.  *Starship Play:* Role-based play, Ship stats, Captain tasks.
        5.  *Comparison:* Narrative-driven vs. tactical-heavy (D&D) or MMO-style (STO).
    *   *Styling:* Star Trek-themed (LCARS-inspired, blue/gold/black palette, sans-serif fonts).
    *   *Generator Approach:* Python (FPDF2 or ReportLab) + Markdown parsing.
    *   *Facts from Corpus:* Terminology check (e.g., "Advantage," "Difficulty," "Momentum").

    *   *(1) Output Format:* PDF (A4, portrait).
    *   *(2) Sections:*
        *   S1: Intro - Philosophy of "Rule of First," era-specific context (TOS to Discovery era).
        *   S2: Mechanics - The 2d20 roll, Attributes (Daring, Control, Insight, etc.), Roles/Departments, Stress management, and Resource Economy.
        *   S3: Creation - Step-by-step Lifepath, Origin-, Talent trees.
        *   S4: Starship - Captain duties, role-based ship combat/exploration.
        *   S5: Comparison - Table/List highlighting narrative depth vs. other systems.
    *   *(3) Styling:* LCARS-inspired UI (rounded corners, color-coded bars). "Exo" or "Roboto" fonts. Bold headers.
    *   *(4) Generator:* `ReportLab` library in Python. Script: Parse Markdown source -> Map headers to LCARS-style headers -> Generate tables for stats/comparison.
    *   *(5) Facts:* Extract specific attribute names, department names, and "Rule of First" terminology from the'star-trek' corpus.<channel|>**Build Plan: Star Trek Adventures Player Guide**

**1. Output Format**
*   **Type:** PDF (A4, Portrait).
*   **Interactivity:** Hyperlinked Table of Contents.

**2. Sections & Content**
*   **Introduction:** Narrative-first philosophy ("Rule of First"); overview of timeline eras (TOS, TNG, DS, Voyager, Picard).
*   **Core Mechanics:** Explanation of the 2d20 system (Success vs. Difficulty); Attributes (Daring, Control, Insight, etc.); Department-based roles (Command, Science, Ops, Medical, Engineering); Stress mechanics; and Momentum/Threat economy.
*   **Character Creation:** Step-by-step Lifepath guide; Origins, Specializations, and Talents.
*   **Starship Play:** Role-based ship operations; Captain-specific tasks; Ship-to-ship combat and exploration.
*   **Why Play?:** Comparative analysis highlighting narrative-driven play vs. D&D (tactical-heavy) and STO (progression-loop).

**3. Styling/Branding**
*   **Visual Language:** LCARS (Library Computer Access and Retrieval System) aesthetic.
*   **Color Palette:** Deep black backgrounds with cyan, gold, purple, and orange accents.
*   **Typography:** Sans-serif fonts (e.g., Roboto or Helvetica) to mimic de-identified computer displays.
*   **UI Elements:** Use rounded-corner headers and horizontal progress bars for stat examples.

**4. Generator Approach**
*   **Library:** Python with `ReportLab` for precise PDF generation and `Markdown2` for content parsing.
*   **Script Outline:**
    1.  Parse raw text from the source corpus.
    2.  Apply LCARS-style CSS-like styling to Markdown headers.
    3.  Generate dynamic tables for Attribute/Department comparisons.
    4.  Inject vector-based Star Trek insignia icons/bars.

**5. Facts to Pull ('star-trek' corpus)**
*   **Terminology:** Ensure use of "Advantage," "Difficulty," "Momentum," and "Stress Points."
*   **Data:** Verify list of Attribute names and Department-specific skills.
*   **Lore:** Confirm era-specific terminology for historical references.
```

<details><summary>Reference — qwen3.6-35b-a3b (substrate)</summary>

```
## Build Plan: "New Player's Guide to Star Trek Adventures" PDF

### (1) Output Format
**PDF** — explicitly requested. Built via **reportlab** (Python) for precise control over layout, styling, and professional typography.

### (2) Structure / Sections

| # | Section | Content |
|---|---------|---------|
| 1 | **Introduction to the Game** | What is Star Trek Adventures? Storytelling & collaborative play; the GM role; **Eras** (Prime Era / TNG Era / Discovery Era / etc.) — terminology from corpus |
| 2 | **Core Mechanics** | **2d20 System** — roll 2d20 + attribute + skill vs. difficulty; **Attributes** (Fitness, Insight, Presence); **Six Departments** (Command, Conn, Engineering, Security, Medicine, Science) with sample skills; **Stress** pool (tied to Fitness); **Momentum / Threat** resource system; **Focuses** (6 per character) |
| 3 | **Character Creation** | **Lifepath** method (one of two creation methods per corpus); species selection; attributes & skills; talents; equipment; example character |
| 4 | **Starship Play** | Departments in space; power allocation; maneuvering; shields & weapons; away missions; the GM's role in starship combat |
| 5 | **Why Play?** | Highlights vs. D&D (collaborative vs. adversarial, exploration vs. combat); highlights vs. Star Trek Online (tabletop intimacy, no grind, infinite scenarios); iconic moments you'll create |

### (3) Branding / Template
- **Color palette**: Starfleet blue (#0077BE), deep space black/navy (#0A0E27), gold accent (#C8A951)
- **Typography**: Clean sans-serif (Helvetica/Roboto) for body; bold headers
- **Style**: Dark section dividers, callout boxes for rules, clean margins — evokes a Starfleet database terminal
- **No official Star Trek logo** (copyright); use text-based branding only

### (4) Generator Approach

**Library**: `reportlab` (Python) — precise PDF generation with:
- `SimpleDocTemplate` for page layout
- `ParagraphStyle` for consistent typography
- `Table` for any tables (e.g., departments/skills)
- `Drawing` for simple decorative elements (section dividers, rule boxes)

**Script outline** (`build_guide.py`):
```python
# 1. Setup: doc dimensions (6x9" trade paperback), margins, styles
# 2. Helper functions: add_section_header(), add_body_text(), add_callout_box(), add_rule_box()
# 3. Section 1: Introduction (storytelling, Eras)
# 4. Section 2: Core Mechanics (2d20, attributes table, departments table, stress, momentum/threat)
# 5. Section 3: Character Creation (lifepath steps, species, example)
# 6. Section 4: Starship Play (departments, power, combat)
# 7. Section 5: Why Play? (comparison tables vs D&D, STO)
# 8. Save to "New_Players_Guide_to_Star_Trek_Adventures.pdf"
```

**Corpus facts to pull in build step** (via `doc_get`):
- Exact attribute names and their skill lists (from `star-trek-core-030`)
- Department descriptions and sample focuses (from `star-trek-core-030`)
- Lifepath creation steps (from `star-trek-core-001` / `star-trek-core-027`)
- Starship power/maneuver mechanics (from `star-trek-core-057`, `star-trek-core-088`)
- Era names and descriptions (from corpus — may need targeted search)
```
</details>

---

## 02-content-star-trek-factions — `build-content`

**Task (binding question):**

> Produce a table titled 'Star Trek Adventures — Major Factions'. One header row, then one row per faction, columns: Faction | Type | Home Quadrant | Stance toward the Federation. Include United Federation of Planets, Klingon Empire, Romulan Star Empire, Cardassian Union, Ferengi Alliance, the Dominion, the Borg Collective, Breen Confederacy.

**DiffusionGemma** (716 tok, 235 tok/s, 3.0s):

```
<|channel>thought
*   Topic: 'Star Trek Adventures — Major Factions'.
    *   Columns: Faction, Type, Home Quadrant, Stance toward the Federation.
    *   Factions to include: United Federation of Planets, Klingon Empire, Romulan Star Empire, Cardassian Union, Ferengi Alliance, the Dominion, the Borg Collective, Breen Confederacy.
    *   Format: Markdown table. No preamble/no code/tools.

    *   *United Federation of Planets*: Type: Interstellar Federation; Quadrant: Alpha; Stance: The Actor.
    *   *Klingon Empire*: Type: Military Empire; Quadrant: Alpha; Stance: Often hostile/Occasional allies.
    *   *Romulan Star Empire*: Type: Galactic Empire; Quadrant: Alpha; Stance: Historically hostile/Suspicious.
    *   *Cardassian Union*: Type: Military Dictatorship; Quadrant: Alpha; Stance: Hostile/Rival.
    *   *Ferengi Alliance*: Type: Mercantile Alliance; Quadrant: Alpha; Stance: Neutral/Trading Partners.
    *   *The Dominion*: Type: Theocratic Empire; Quadrant: Gamma; Stance: Expansionist/Hostile.
    *   *The Borg Collective*: Type: Cybernetic Hive Mind; Quadrant: Delta; Stance: Aggressive/Assimilatory.
    *   *Breen Confederacy*: Type: Military Confederacy; Quadrant: Alpha (usually border); Stance: Hostile.

    | Faction | Type | Home Quadrant | Stance toward the Federation |
    | :--- | :--- | :--- | :--- |
    | United Federation of Planets | Interstellar Federation | Alpha | The Actor |
    | Klingon Empire | Military Empire | Alpha | Occasional Allies / Rivals |
    | Romulan Star Empire | Galactic Empire | Alpha | Hostile / Suspicious |
    | Cardassian Union | Military Dictatorship | Alpha | Hostile / Rivals |
    | Ferengi Alliance | Mercantile Alliance | Alpha | Neutral / Trading Partners |
    | The Dominion | Theocratic Empire | Gamma | Expansionist / Hostile |
    | The Borg Collective | Cybernetic Hive Mind | Delta | Aggressive Assimilators |
    | Breen Confederacy | Military Confederacy | Alpha | Hostile |

    *   Title: Star Trek Adventures — Major Factions
    *   Header row: Yes.
    *   One row per faction: Yes.
    *   No preamble: Yes.<channel|># Star Trek Adventures — Major Factions

| Faction | Type | Home Quadrant | Stance toward the Federation |
| :--- | :--- | :--- | :--- |
| United Federation of Planets | Interstellar Federation | Alpha | The Actor |
| Klingon Empire | Military Empire | Alpha | Occasional Allies / Rivals |
| Romulan Star Empire | Galactic Empire | Alpha | Hostile / Suspicious |
| Cardassian Union | Military Dictatorship | Alpha | Hostile / Rivals |
| Ferengi Alliance | Mercantile Alliance | Alpha | Neutral / Trading Partners |
| The Dominion | Theocratic Empire | Gamma | Expansionist / Hostile |
| The Borg Collective | Cybernetic Hive Mind | Delta | Aggressive Assimilators |
| Breen Confederacy | Military Confederacy | Alpha | Hostile |
```

<details><summary>Reference — qwen3.6-35b-a3b (substrate)</summary>

```
| Faction | Type | Home Quadrant | Stance toward the Federation |
|---|---|---|---|
| United Federation of Planets | Government / Alliance | Alpha & Beta Quadrants | N/A — the Federation itself |
| Klingon Empire | Empire / Military | Beta Quadrant | Former enemy; now ally after Khitomer Accords |
| Romulan Star Empire | Empire / Theocracy | Alpha Quadrant | Former enemy; joined Alpha Quadrant alliance during Dominion War |
| Cardassian Union | Union / Totalitarian State | Alpha Quadrant | Former enemy (war 2347–2366); later ally |
| Ferengi Alliance | Alliance / Capitalist Economy | Gamma Quadrant | Neutral; focused on commerce; later joined alliance |
| Dominion | Dominion / Military Dictatorship | Gamma Quadrant | Hostile; invaded in Dominion War (2373–2375) |
| Borg Collective | Collective / Hive Mind | Delta Quadrant | Hostile; gravest threat the Galaxy has faced |
| Breen Confederacy | Confederacy / Theocracy | Gamma Quadrant | Former enemy; allied with Dominion; later neutral |

_(Reference note: the substrate's build stage sourced these from the Star Trek Adventures 2e Core Rulebook corpus via `doc_search`. The Ferengi/Breen "Gamma Quadrant" entries are the substrate model's own output, faithfully reproduced — Ferengi/Breen are canonically Alpha-Quadrant, so this reference is itself imperfect, which makes it a fair head-to-head.)_
```
</details>

---

## 03-toolcall-weather — `tool-call`

**Task (binding question):**

> What's the weather in Tokyo right now?

**DiffusionGemma** (63 tok, 102 tok/s, 0.6s):

```
<|channel>thought
The user is asking for the current weather in Tokyo.
I should use the `get_current_weather` tool for this request.
The tool requires a `city` parameter, which will be "Tokyo".<channel|><|tool_call>call:get_current_weather{city: "Tokyo"}<tool_call|>
```

<details><summary>Reference — (capability probe — no substrate reference) (substrate)</summary>

```
Expected: a structured tool call, e.g. <|tool_call>call:get_current_weather{city: "Tokyo"}<tool_call|>. This probe corrects an earlier WRONG claim that DiffusionGemma 'is not a tool-calling model' — it has native function calling (Google model card) and emits the tool-call special tokens through the visual-server harness with tools merely inlined.
```
</details>
