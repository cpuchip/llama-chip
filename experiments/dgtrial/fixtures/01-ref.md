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
