# Global Rules

- Avoid use of metaphor, idioms, metonymy.
- Claude does not use idioms or jargon metaphors.
- Claude does not use intensifiers as credibility markers.
- Claude begins responses with the statement it wishes to make.
- Never silence lint/type errors: no suppression comments, warning-config edits, or escape hatches. Precedent doesn't authorize.
- Keep Bash calls simple and verifiable; compound/loop shell syntax triggers permission prompts and things grind to a halt.
- Renaming a symbol? Look for IDE based mcp rename_refactoring. Intelligently renames a symbol vs dumb find and replace.

## Code Quality Rules

- For mechanical fixes, run the formatter/linter. Don't hand-edit, including when surfaced as IDE diagnostics.
- Optimize for legibility, predictability, and low cognitive load. Resist clever local fixes that shadow or override codebase conventions.
- Names reveal intent. A reader shouldn't need to read the body to know what a function does.
- Prefer deep modules over shallow ones (Ousterhout): a function earns its place only when its interface is simpler than the body it hides. Don't write macros, functions whose body is a single expression or stdlib call and whose name restates it; write the expression inline.
- When you find a code smell, look for the structural cause one layer deeper. The fix usually lives there, not where the smell shows up.

## Python

- Follow the Functional Core, Imperative Shell pattern: pure logic at the core, side effects pushed to the boundary.
- Inject dependencies through the constructor, typing strategies as a `Protocol` and passing configuration as the concrete Pydantic settings class.
- I'm using `uv`, not `pip`. Run tools with `uv run`.
- Target Python 3.12+ and use its syntax: `X | None`, built-in generics, PEP 695 type parameters.
- No assert statements in production code.
- Raise built-in exception types unless a custom exception handler needs its own type. No bare or broad `except`, and don't swallow errors silently.
- Matches should be provably exhaustive. Even if type checker says it's unreachable. End with `case _:` calling `typing.assert_never`, or raising an exception when that reads clearer.
- (Pydantic OR Dataclass) prefer over (Namedtuple) prefer over (TypedDict OR dict[str, object]) prefer over (dict).
- When working with a domain object that passes through stages where some fields are only valid at certain stages, design a type (Pydantic or dataclass) for each stage such that invariant violations are impossible to represent.
- Parse, don't validate: parse data at the boundary into a type that records what was checked. Don't add a fallback value to satisfy a type, because the fallback shows the type is wrong.
- No `_` prefixes on code you write. That includes functions, classes, variables
- Lowercase enum member names.
- Module level constants should not be your first choice.
- Always use `ruff check` with `--fix`.

## Planning

- Never explicitly write examples. Only write the main ideas, and concerns.
- Avoid concrete references (file names, line numbers, derived statistics); a single rename forces updates everywhere they're cited. Refer to concepts instead.
- Don't rationalize exceptions to written rules or approved plans; local circumstances always feel like they justify it. Surface the conflict instead.

## Github

- Subscribe to your PRs.
- Response style applies to PRs, comments.
- A Git hook reminder about uncommitted or unpushed changes needs no reply in the chat. During a discussion, do not add a paragraph that says the changes are not committed or not pushed.
