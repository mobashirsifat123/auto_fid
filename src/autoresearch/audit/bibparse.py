"""Strict BibTeX and ``thebibliography`` parsing for the trust-audit harness.

Implements order 3 of ``research/trust_audit/DESIGN.md`` (§6.3, §11): a
minimal but strict parser that converts a ``.bib`` file or an embedded LaTeX
``\\begin{thebibliography}`` environment into normalized :class:`BibEntry`
records for the citation-resolution metric (§5.1).

Fail-closed contract (§4.3 asymmetry — parse errors are artifact-side data):

* A malformed entry is **data**, never a crash: it is reported as a
  :class:`BibParseError` with a machine-readable ``reason`` (and the entry
  key when recoverable) so the citations metric can count it against the
  audited output as ``UNRESOLVABLE/unparseable``. The parser never guesses a
  field value it could not read.
* An entry using ``crossref`` is kept with its own fields and recorded as
  ``crossref_not_expanded``: inline expansion is not attempted (a partially
  inherited entry would fabricate fields).
* Unknown string macros resolve to the empty string and are recorded as
  ``unknown_macro`` errors; month macros (``jan`` … ``dec``) and file-local
  ``@string`` definitions are resolved.
* In ``thebibliography`` items, only fields with a deterministic delimiter
  are extracted (quoted title, trailing year, DOI/arXiv patterns); author
  parsing reuses the BibTeX name rules on the text preceding the title.
  An item without a delimited title records ``title_not_delimited`` and
  yields an empty title rather than guessing.

Deviations from DESIGN.md §6 (recorded rationale): ``BibEntry``,
``BibAuthor``, and ``BibParseError`` are defined in this module rather than
in ``contract.py`` (the §6 table lists ``BibEntry`` under contract) because
``contract.py`` imports this module for adapter-side bibliography
collection; defining the record types here keeps the import graph acyclic,
and ``contract.py`` re-exports them unchanged so both import paths work for
the metric modules built in parallel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from autoresearch.papers.metadata_sources import normalize_arxiv_id, normalize_doi

_SNIPPET_LIMIT = 200

_MONTHS = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
}

_ENTRY_HEAD = re.compile(r"@([A-Za-z]+)\s*(\{|\()")
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
_NUMBER = re.compile(r"[0-9]+")
_YEAR = re.compile(r"\b(\d{4})\b")
_ARXIV_ID = re.compile(r"^[0-9]{4}\.[0-9]{4,5}$|^[a-z\-]+/[0-9]{7}$")

_ACCENT = re.compile(r"\\[`'^\"~=.]\s*([A-Za-z])")
_ESCAPED_SPECIAL = re.compile(r"\\([%&#$_])")
_COMMAND = re.compile(r"\\[A-Za-z]+\s*")

_BIB_ENV = re.compile(
    r"\\begin\{thebibliography\}\s*(?:\{[^}]*\})?(.*?)\\end\{thebibliography\}", re.DOTALL
)
_BIBITEM = re.compile(r"\\bibitem(?:\s*\[[^\]]*\])?\s*\{([^}]*)\}")
_QUOTED_TITLE = re.compile(r"``([^']+)''|\"([^\"]+)\"|“([^”]+)”")
_DOI_IN_TEXT = re.compile(r"\b(10\.\d{4,9}/[^\s\"”<>]+)")
_ARXIV_IN_TEXT = re.compile(
    r"\b(?:arxiv\s*[:\s]|arxiv\.org/(?:abs|pdf)/)"
    r"([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+/[0-9]{7})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BibAuthor:
    """One person record from a bibliography author list ("others" allowed)."""

    family: str
    given: str


@dataclass(frozen=True)
class BibEntry:
    """One normalized bibliography entry (§5.1 field set).

    ``doi``/``arxiv_id`` are normalized via ``papers.metadata_sources``;
    ``year`` is ``None`` when absent or unparsable (missing data is never
    fabricated). ``title``/``venue`` carry LaTeX-normalized text.
    """

    key: str
    entry_type: str
    title: str
    authors: tuple[BibAuthor, ...] = ()
    year: int | None = None
    venue: str = ""
    doi: str | None = None
    arxiv_id: str | None = None


@dataclass(frozen=True)
class BibParseError:
    """One per-entry parse failure: artifact-side data, never a crash."""

    key: str | None
    reason: str
    detail: str


@dataclass(frozen=True)
class BibParseResult:
    entries: tuple[BibEntry, ...] = ()
    errors: tuple[BibParseError, ...] = ()
    source: str = "bibtex"
    entry_count: int = 0
    error_count: int = field(default=0)

    def __post_init__(self) -> None:
        # Convenience counters so metric code never recomputes counts from
        # the tuples (rates must follow recorded counts — report.py §8.3).
        object.__setattr__(self, "entry_count", len(self.entries))
        object.__setattr__(self, "error_count", len(self.errors))


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def parse_bibtex(text: str) -> BibParseResult:
    """Parse strict BibTeX source into entries + per-entry errors."""
    entries: list[BibEntry] = []
    errors: list[BibParseError] = []
    macros: dict[str, str] = {}
    seen: set[str] = set()
    position = 0
    while True:
        at = text.find("@", position)
        if at == -1:
            break
        head = _ENTRY_HEAD.match(text, at)
        if head is None:
            # Not an entry head (e.g. "@" inside a comment line): skip.
            position = at + 1
            continue
        entry_type = head.group(1).lower()
        opener = head.group(2)
        body_start = head.end()
        end, balanced = _find_entry_end(text, body_start, opener)
        if not balanced:
            errors.append(BibParseError(
                key=None, reason="truncated_entry",
                detail=f"unterminated @{entry_type} entry at offset {at}: {_snippet(text[at:])}",
            ))
            break
        body = text[body_start:end - 1]
        position = end
        if entry_type in {"comment", "preamble"}:
            continue
        if entry_type == "string":
            fields, field_errors = _parse_fields(body, macros, key=None)
            errors.extend(field_errors)
            if len(fields) == 1:
                (name, value), = fields.items()
                macros[name] = value
            elif not field_errors:
                errors.append(BibParseError(
                    key=None, reason="invalid_string_definition",
                    detail=f"@string at offset {at} does not define exactly one macro: {_snippet(body)}",
                ))
            continue
        _parse_entry(entry_type, body, at, macros, seen, entries, errors)
    return BibParseResult(tuple(entries), tuple(errors), "bibtex")


def parse_thebibliography(text: str) -> BibParseResult:
    """Extract ``\\bibitem`` entries from LaTeX source (any nesting depth)."""
    entries: list[BibEntry] = []
    errors: list[BibParseError] = []
    seen: set[str] = set()
    for body in _BIB_ENV.findall(text):
        items = list(_BIBITEM.finditer(body))
        if not items and body.strip():
            errors.append(BibParseError(
                key=None, reason="no_bibitem",
                detail=f"thebibliography environment has no \\bibitem entries: {_snippet(body)}",
            ))
            continue
        for index, match in enumerate(items):
            key = match.group(1).strip()
            start = match.end()
            end = items[index + 1].start() if index + 1 < len(items) else len(body)
            raw_item = body[start:end]
            if not key:
                errors.append(BibParseError(
                    key=None, reason="missing_key",
                    detail=f"\\bibitem without a key: {_snippet(raw_item)}",
                ))
                continue
            if key in seen:
                errors.append(BibParseError(
                    key=key, reason="duplicate_key",
                    detail=f"bibliography key '{key}' already defined; second entry skipped",
                ))
                continue
            seen.add(key)
            entry, item_errors = _build_thebibliography_item(key, raw_item)
            entries.append(entry)
            errors.extend(item_errors)
    return BibParseResult(tuple(entries), tuple(errors), "thebibliography")


def has_thebibliography(text: str) -> bool:
    return _BIB_ENV.search(text) is not None


# ----------------------------------------------------------------------
# BibTeX entry parsing
# ----------------------------------------------------------------------


def _find_entry_end(text: str, start: int, opener: str) -> tuple[int, bool]:
    """Return (index_after_closer, balanced) for an entry body."""
    if opener == "{":
        depth = 1
        index = start
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index + 1, True
            index += 1
        return len(text), False
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == ")" and depth == 0:
            return index + 1, True
        index += 1
    return len(text), False


def _parse_entry(
    entry_type: str, body: str, offset: int, macros: dict[str, str],
    seen: set[str], entries: list[BibEntry], errors: list[BibParseError],
) -> None:
    comma = body.find(",")
    if comma == -1:
        key, rest = body.strip(), ""
    else:
        key, rest = body[:comma].strip(), body[comma + 1:]
    if not key:
        errors.append(BibParseError(
            key=None, reason="missing_key",
            detail=f"entry at offset {offset} has no key: {_snippet(body)}",
        ))
        return
    if key in seen:
        errors.append(BibParseError(
            key=key, reason="duplicate_key",
            detail=f"key '{key}' already defined; second entry skipped: {_snippet(body)}",
        ))
        return
    seen.add(key)
    fields, field_errors = _parse_fields(rest, macros, key=key)
    errors.extend(field_errors)
    entries.append(_build_entry(key, entry_type, fields))
    if "crossref" in fields:
        errors.append(BibParseError(
            key=key, reason="crossref_not_expanded",
            detail=f"entry '{key}' uses crossref '{fields['crossref']}'; inline expansion is not supported",
        ))


def _parse_fields(
    body: str, macros: dict[str, str], key: str | None,
) -> tuple[dict[str, str], list[BibParseError]]:
    """Parse ``name = value`` pairs; field names are lowercased."""
    fields: dict[str, str] = {}
    errors: list[BibParseError] = []
    index = 0
    length = len(body)
    while index < length:
        while index < length and body[index] in " \t\r\n,":
            index += 1
        if index >= length:
            break
        name_match = _FIELD_NAME.match(body, index)
        if name_match is None:
            stop = body.find(",", index)
            stop = length if stop == -1 else stop
            errors.append(BibParseError(
                key=key, reason="invalid_field",
                detail=f"unparsable field syntax: {_snippet(body[index:stop])}",
            ))
            index = stop + 1
            continue
        name = name_match.group(0).lower()
        index = name_match.end()
        while index < length and body[index] in " \t\r\n":
            index += 1
        if index >= length or body[index] != "=":
            stop = body.find(",", index)
            stop = length if stop == -1 else stop
            errors.append(BibParseError(
                key=key, reason="invalid_field",
                detail=f"field '{name}' is not followed by '=': {_snippet(body[index:stop])}",
            ))
            index = stop + 1
            continue
        index += 1
        value, index, error = _parse_value(body, index, macros, key, name)
        if error is not None:
            errors.append(error)
            if not value:
                continue  # unreadable value: skip to the recorded failure point
        fields[name] = value
    return fields, errors


def _parse_value(
    body: str, index: int, macros: dict[str, str], key: str | None, field: str,
) -> tuple[str, int, BibParseError | None]:
    """Parse one field value with ``#`` concatenation and macro resolution."""
    parts: list[str] = []
    length = len(body)
    error: BibParseError | None = None
    while True:
        while index < length and body[index] in " \t\r\n":
            index += 1
        if index >= length:
            if not parts:
                return "", index, BibParseError(
                    key=key, reason="missing_value",
                    detail=f"field '{field}' of entry '{key}' has no value",
                )
            return "".join(parts), index, None
        char = body[index]
        if char == "{":
            end, balanced = _find_entry_end(body, index + 1, "{")
            if not balanced:
                return "".join(parts), length, BibParseError(
                    key=key, reason="unterminated_value",
                    detail=f"field '{field}' of entry '{key}' has unbalanced braces",
                )
            parts.append(body[index + 1:end - 1])
            index = end
        elif char == '"':
            closing = _find_quote_end(body, index + 1)
            if closing == -1:
                return "".join(parts), length, BibParseError(
                    key=key, reason="unterminated_value",
                    detail=f"field '{field}' of entry '{key}' has an unterminated quoted value",
                )
            parts.append(body[index + 1:closing])
            index = closing + 1
        else:
            number = _NUMBER.match(body, index)
            macro = _FIELD_NAME.match(body, index)
            if number is not None:
                parts.append(number.group(0))
                index = number.end()
            elif macro is not None:
                resolved = macros.get(macro.group(0).lower())
                if resolved is None:
                    resolved = _MONTHS.get(macro.group(0).lower())
                if resolved is None:
                    if error is None:
                        error = BibParseError(
                            key=key, reason="unknown_macro",
                            detail=f"field '{field}' of entry '{key}' references undefined macro '{macro.group(0)}'",
                        )
                    resolved = ""
                parts.append(resolved)
                index = macro.end()
            else:
                stop = body.find(",", index)
                stop = length if stop == -1 else stop
                reason = "missing_value" if char == "," else "invalid_value"
                return "".join(parts), stop, BibParseError(
                    key=key, reason=reason,
                    detail=f"field '{field}' of entry '{key}' has an unreadable value: {_snippet(body[index:stop])}",
                )
        while index < length and body[index] in " \t\r\n":
            index += 1
        if index < length and body[index] == "#":
            index += 1
            continue
        return "".join(parts), index, error


def _find_quote_end(body: str, start: int) -> int:
    """Find the closing quote at brace depth zero (braces protect quotes)."""
    depth = 0
    index = start
    while index < len(body):
        char = body[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == '"' and depth == 0:
            return index
        index += 1
    return -1


def _build_entry(key: str, entry_type: str, fields: dict[str, str]) -> BibEntry:
    venue = ""
    if entry_type in {"inproceedings", "conference", "proceedings"}:
        preference = ("booktitle", "journal", "publisher")
    else:
        preference = ("journal", "booktitle", "publisher")
    for candidate in preference:
        if fields.get(candidate, "").strip():
            venue = _latex_text(fields[candidate])
            break
    return BibEntry(
        key=key,
        entry_type=entry_type,
        title=_latex_text(fields.get("title", "")),
        authors=_split_authors(fields.get("author", "")),
        year=_parse_year(fields.get("year", "")),
        venue=venue,
        doi=normalize_doi(fields.get("doi", "") or fields.get("url", "")),
        arxiv_id=_arxiv_from_fields(fields),
    )


def _arxiv_from_fields(fields: dict[str, str]) -> str | None:
    eprint = fields.get("eprint", "").strip()
    prefix = fields.get("archiveprefix", "") or fields.get("eprinttype", "")
    if eprint:
        candidate = normalize_arxiv_id(eprint)
        if candidate and ("arxiv" in prefix.lower() or _ARXIV_ID.match(candidate)):
            return candidate
    url = fields.get("url", "")
    if "arxiv.org" in url.lower():
        return normalize_arxiv_id(url)
    return None


def _parse_year(value: str) -> int | None:
    match = _YEAR.search(value)
    if match is None:
        return None
    year = int(match.group(1))
    return year if 1000 <= year <= 3000 else None


# ----------------------------------------------------------------------
# thebibliography item parsing
# ----------------------------------------------------------------------


def _build_thebibliography_item(
    key: str, raw_item: str,
) -> tuple[BibEntry, list[BibParseError]]:
    errors: list[BibParseError] = []
    title_match = _QUOTED_TITLE.search(raw_item)
    title = ""
    authors: tuple[BibAuthor, ...] = ()
    if title_match is None:
        errors.append(BibParseError(
            key=key, reason="title_not_delimited",
            detail=f"item '{key}' has no quoted title (strict extraction refuses to guess): {_snippet(raw_item)}",
        ))
    else:
        title = _latex_text(next(group for group in title_match.groups() if group is not None))
        # Trailing commas/semicolons are citation punctuation (American
        # "Title," style), not part of the title; periods are kept.
        title = title.rstrip(",;: ").strip()
        prefix = raw_item[:title_match.start()]
        prefix = re.sub(r"[\s,;:]+$", "", prefix).strip()
        if prefix:
            authors = _split_authors(prefix)
    year = None
    text_for_year = _ARXIV_IN_TEXT.sub(" ", _DOI_IN_TEXT.sub(" ", raw_item))
    years = list(_YEAR.finditer(text_for_year))
    if years:
        year = _parse_year(years[-1].group(0))
    doi_match = _DOI_IN_TEXT.search(raw_item)
    arxiv_match = _ARXIV_IN_TEXT.search(raw_item)
    return BibEntry(
        key=key,
        entry_type="thebibliography_item",
        title=title,
        authors=authors,
        year=year,
        venue="",
        doi=normalize_doi(doi_match.group(1)) if doi_match else None,
        arxiv_id=normalize_arxiv_id(arxiv_match.group(1)) if arxiv_match else None,
    ), errors


# ----------------------------------------------------------------------
# Author names and LaTeX normalization
# ----------------------------------------------------------------------


def _split_authors(value: str) -> tuple[BibAuthor, ...]:
    if not value.strip():
        return ()
    authors: list[BibAuthor] = []
    for part in _split_on_and(value):
        name = part.strip()
        if not name:
            continue
        if _latex_text(name).strip().lower() == "others":
            authors.append(BibAuthor(family="others", given=""))
            continue
        authors.append(_parse_name(name))
    return tuple(authors)


def _split_on_and(value: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    length = len(value)
    while index < length:
        char = value[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif depth == 0 and value[index:index + 5].lower() == " and ":
            parts.append(value[start:index])
            index += 5
            start = index
            continue
        index += 1
    parts.append(value[start:])
    return parts


def _parse_name(name: str) -> BibAuthor:
    comma = _depth_zero_index(name, ",")
    if comma != -1:
        family = _latex_text(name[:comma])
        given = _latex_text(name[comma + 1:])
        return BibAuthor(family=family, given=given)
    tokens = _name_tokens(name)
    braced = [token for token in tokens if token.startswith("{") and token.endswith("}")]
    if braced:
        family_token = braced[-1]
        remaining = [token for token in tokens if token != family_token]
        return BibAuthor(family=_latex_text(family_token), given=_latex_text(" ".join(remaining)))
    if tokens:
        return BibAuthor(family=_latex_text(tokens[-1]), given=_latex_text(" ".join(tokens[:-1])))
    return BibAuthor(family="", given="")


def _depth_zero_index(value: str, char: str) -> int:
    depth = 0
    for index, current in enumerate(value):
        if current == "{":
            depth += 1
        elif current == "}":
            depth = max(0, depth - 1)
        elif current == char and depth == 0:
            return index
    return -1


def _name_tokens(name: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    for char in name:
        if char == "{":
            depth += 1
            current.append(char)
        elif char == "}":
            depth = max(0, depth - 1)
            current.append(char)
        elif char in " \t\r\n" and depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _latex_text(value: str) -> str:
    """Normalize a LaTeX field value: drop commands/braces, keep characters."""
    text = _ACCENT.sub(r"\1", value)
    text = _ESCAPED_SPECIAL.sub(r"\1", text)
    text = _COMMAND.sub(" ", text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    text = text.replace("\\", "")
    return " ".join(text.split())


def _snippet(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= _SNIPPET_LIMIT:
        return normalized
    return normalized[:_SNIPPET_LIMIT] + "…"
