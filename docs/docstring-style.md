# Docstring style guide

Docstrings use the NumPy style.
mkdocstrings renders names, signatures, type annotations and defaults.
Docstrings hold only information that these do not show.

## When to write a docstring

1. A docstring exists only for behavior that names, signatures and annotations cannot show.
   A docstring that only explains a name shows that the name or the structure needs to change.
2. Observable behavior appears only when it affects how the caller uses the construct.
   The internal mechanism stays out.
3. An example replaces prose when the prose needs many words to describe a behavior.

| Rule | Avoid | Use |
| --- | --- | --- |
| 1 | A send with a node and an arg. | Runs `node` once with `arg` as its input in place of the graph state. |
| 2 | Calls deepcopy on initial inside make_initial. | Each run starts from a separate copy of `initial`. |

## Grammar

4. Sentences use the third-person indicative mood.
5. Sentences address no reader. The second person is absent.
6. Sentences are affirmative. Negations are absent.
7. Statements are definite. Modal verbs for hedging and conditional clauses are absent.
8. Sentences use the active voice.
9. Verbs use the simple present tense.
10. Verbs carry the action. Nominalizations are absent.
11. Each sentence starts with its real subject. Expletive constructions are absent.

| Rule | Avoid | Use |
| --- | --- | --- |
| 4 | Merge each write into a state field. | Merges each write into a state field. |
| 5 | Set update_type if you need it. | `update_type` is necessary when a write and the field have different types. |
| 6 | Adds targets and does not replace the static edges. | Adds targets to the static edges. |
| 7 | May return the stored value. | Returns the stored value. |
| 8 | Each write is validated by the channel. | The channel validates each write. |
| 9 | The graph will call `function`. | The graph calls `function`. |
| 10 | Performs validation of each write. | Validates each write. |
| 11 | There is a check that rejects unknown nodes. | `goto` rejects unknown nodes. |

## Sentence structure

12. Each sentence has one independent clause and one idea.
    Dashes, colons, semicolons and parentheses do not join or insert clauses in prose.
    Short sentences keep each pronoun tied to one noun.
13. Items in a list share one grammatical form.
14. The first line states the main behavior. Later lines add details in order of importance.

| Rule | Avoid | Use |
| --- | --- | --- |
| 12 | Validates each write (the field's type by default); then merges it. | Validates each write against `update_type`. The default `update_type` is the field's type. |
| 13 | Accepts a node name, sending to a node, or you can end the run. | Accepts a node name, a `Send` or `Endpoint.end`. |

## Word choice

15. Language is literal. Metaphors, idioms, personification, minimizers and intensifiers are absent.
16. Verbs name the exact action. Generic verbs are absent.
17. Each concept has one name throughout the text.
    Code elements appear with their exact identifiers in backticks.
18. Quantities are exact and include units.
19. The docstring describes current behavior only. Temporal references and history are absent.
20. The docstring states the behavior directly. Self-reference to the documented construct is absent.

| Rule | Avoid | Use |
| --- | --- | --- |
| 15 | Simply hands off control to review. | The graph runs `review` after this node. |
| 16 | Handles writes and manages the value. | Validates each write and merges the write into the stored value. |
| 17 | Validates each update, then merges the write into the value. | Validates each write, then merges the write into the field. |
| 18 | Accepts a small message. | Accepts a message of at most 64 KiB. |
| 19 | Now validates each write. | Validates each write. |
| 20 | This class merges each write. | Merges each write. |
