"""Conservative, offline review rules shared by generation and import paths.

These rules quarantine contradictions; they do not establish negative facts.
Only a relation's own description is inspected, never unrelated page text.
"""
import copy
import re


RULE_VERSION = "relation_local_contradictions_v2"
_NEGATION_RULE_VERSION = "requires_explicit_negation_v1"
_TYPE_CHOICE_RULE_VERSION = "relation_explicit_type_choice_v1"
_NEGATED_NEED = re.compile(
    r"不\s*(?:需要|需|须|依赖|要求)|无\s*[需须]|不必\s*(?:需要|依赖)"
    r"|\b(?:does|do|did|will|would|should|must)\s+not\s+(?:need|require|depend\s+on)\b"
    r"|\b(?:doesn't|don't|didn't)\s+(?:need|require|depend\s+on)\b"
    r"|\b(?:needs?|requires?)\s+no\b|\bwithout\s+(?:needing|requiring)\b",
    re.IGNORECASE,
)
_UNCERTAIN_OR_DOUBLE_NEGATION = re.compile(
    r"(?:不能|不可|不可以|不可能|并非|并不是|不是|并不|并不能|未必|不一定|"
    r"可能|也许|或许|是否|能否|假如|如果|若|不见得|没有理由|无理由)\s*(?:\w{0,12}\s*)?$"
    r"|(?:not\s+true\s+that|incorrect\s+that|maybe|perhaps|possibly|whether|if)\b[^,;.!?]*$",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r"[，,。；;！!？?\n]")
_SENTENCE_BREAK = re.compile(r"[。；;！!？?\n]")
_META_DENIAL = re.compile(r"(?:说法|判断|断言|表述).{0,8}(?:不正确|错误|不成立)|(?:不正确|错误).{0,8}(?:说法|判断|断言|表述)")
_CONDITIONAL_SCOPE = re.compile(r"如果|假如|除非|当.+时|在.+(?:时|条件下)|\b(?:if|when|unless)\b", re.IGNORECASE)

# Kept local to avoid importing the builder (which imports this module). Only
# the generator's established semantic types qualify; arbitrary uppercase words
# and speculative new labels do not establish a contradiction.
_TYPE_CHOICE_RELATIONS = frozenset({
    "IS_A", "PART_OF", "HAS_PARAMETER", "CONTROLS", "DRIVES", "CONNECTS_TO",
    "FLOWS_TO", "IMPLEMENTS", "REQUIRES", "PREREQUISITE_OF", "RELATED_TO",
})
_TYPE_PATTERN = "|".join(sorted(_TYPE_CHOICE_RELATIONS, key=len, reverse=True))
_EXPLICIT_TYPE_CHOICE = re.compile(
    r"(?:这里|此处|本处|最终)\s*(?:明确\s*)?(?:选择|选用|采用|确定为)\s*"
    r"(?:关系(?:类型)?\s*(?:为|是)?\s*)?(?P<chosen>" + _TYPE_PATTERN + r")(?![A-Z0-9_])",
    re.IGNORECASE,
)
_RELATION_TYPE_MENTION = re.compile(r"(?<![A-Z0-9_])(?:" + _TYPE_PATTERN + r")(?![A-Z0-9_])", re.IGNORECASE)
_TYPE_CHOICE_UNSAFE = re.compile(
    r"不|未|无|没|否|并非|若|如果|假如|除非|当.+时|在.+条件|"
    r"可能|也许|或许|似乎|疑似|建议|倾向|考虑|待定|暂定|取决于|或者|或是|可选|可以|更正|"
    r"[？?]|\b(?:if|when|unless|not|no|maybe|perhaps|possibly|whether|might|could)\b",
    re.IGNORECASE,
)
_TYPE_CHOICE_EXAMPLE = re.compile(r"例如|比如|举例|示例|案例|\b(?:example|e\.g\.)", re.IGNORECASE)
_TYPE_CHOICE_REPORTED = re.compile(
    r"有人|声称|据称|原话|原文|转述|说法|假设|引述|引用|对比|比较|另一|其他关系|别的|上例|下例|样例|模板|演示"
    r"|\b(?:quote|quoted|says|said|claims|reported|according\s+to)\b", re.IGNORECASE,
)
_TYPE_CHOICE_SENTENCES = re.compile(r"[^。；;！!？?\n]+(?:[。；;！!？?\n]|$)")


def _inside_quoted_text(description, position):
    prefix = description[:position]
    for opening, closing in (("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』")):
        if prefix.rfind(opening) > prefix.rfind(closing):
            return True
    for delimiter in ('"', "'", "```"):
        if prefix.count(delimiter) % 2:
            return True
    return False


def _explicit_type_conflict(record):
    """Find an unconditional self-selected type inconsistent with pending data.

    Earlier discussion may be conditional, as long as a later separate sentence
    clearly resolves it. A later relation mention or reservation makes the
    candidate inconclusive. Quoted examples and teacher decisions stay intact.
    This only quarantines the record; it never chooses the replacement type.
    """
    if record.get("review_status") != "pending":
        return None
    actual = str(record.get("relation", "")).strip().upper()
    description = record.get("description")
    if actual not in _TYPE_CHOICE_RELATIONS or not isinstance(description, str):
        return None
    if _TYPE_CHOICE_EXAMPLE.search(description):
        return None
    candidates = []
    for sentence in _TYPE_CHOICE_SENTENCES.finditer(description):
        text = sentence.group(0)
        if (_TYPE_CHOICE_UNSAFE.search(text) or _TYPE_CHOICE_REPORTED.search(text)
                or re.search(r'[“”‘’"\'「」『』]', text)):
            continue
        for match in _EXPLICIT_TYPE_CHOICE.finditer(text):
            start = sentence.start() + match.start()
            end = sentence.start() + match.end()
            if _inside_quoted_text(description, start):
                continue
            earlier_text = description[:sentence.start()].rstrip()
            if earlier_text.endswith((":", "：")):
                previous_sentence = re.split(r"[。；;！!？?\n]", earlier_text)[-1]
                if _TYPE_CHOICE_UNSAFE.search(previous_sentence) or _TYPE_CHOICE_REPORTED.search(previous_sentence):
                    continue
            suffix_types = {item.group(0).upper() for item in _RELATION_TYPE_MENTION.finditer(text[match.end():])}
            if suffix_types - {match.group("chosen").upper()}:
                continue
            tail = description[sentence.end():]
            if _TYPE_CHOICE_UNSAFE.search(tail) or _RELATION_TYPE_MENTION.search(tail):
                continue
            candidates.append({"chosen": match.group("chosen").upper(),
                               "matched_text": match.group(0), "description_span": [start, end],
                               "description_clause": text.strip()})
    if not candidates:
        return None
    chosen = candidates[-1]
    if chosen["chosen"] == actual:
        return None
    return {"actual_relation": actual, "explicitly_selected_relation": chosen["chosen"],
            **{key: value for key, value in chosen.items() if key != "chosen"}}


def _explicit_negation(record):
    description = record.get("description")
    if not isinstance(description, str) or not description.strip():
        return None
    # Quoted claims, questions and uncertainty are left for a person; a regex
    # cannot safely resolve their scope or infer an affirmative dependency.
    if re.search(r"[？?]", description) or _META_DENIAL.search(description):
        return None
    target = record.get("target")
    target = target.strip() if isinstance(target, str) else ""
    for match in _NEGATED_NEED.finditer(description):
        sentence_breaks = list(_SENTENCE_BREAK.finditer(description, 0, match.start()))
        sentence_left = sentence_breaks[-1].end() if sentence_breaks else 0
        sentence_end = _SENTENCE_BREAK.search(description, match.end())
        sentence_right = sentence_end.start() if sentence_end else len(description)
        if _CONDITIONAL_SCOPE.search(description[sentence_left:sentence_right]):
            continue
        left_breaks = list(_CLAUSE_BREAK.finditer(description, 0, match.start()))
        left = left_breaks[-1].end() if left_breaks else 0
        right_break = _CLAUSE_BREAK.search(description, match.end())
        right = right_break.start() if right_break else len(description)
        prefix = description[left:match.start()]
        if _UNCERTAIN_OR_DOUBLE_NEGATION.search(prefix):
            continue
        suffix = description[match.end():right]
        if re.match(r"\s*(?:不|未|无|no\b|not\b)", suffix, re.IGNORECASE):
            continue
        # If the target is explicitly named somewhere, require it in the
        # negated object span. "A needn't preheat, but requires B" is not a
        # denial of B. Missing target text remains a relation-local claim.
        if target and target.casefold() in description.casefold() and target.casefold() not in suffix.casefold():
            continue
        return {"matched_text": match.group(0), "description_span": [match.start(), match.end()],
                "description_clause": description[left:right]}
    return None


def partition_relationships(relations):
    """Return usable records and auditable pending items, without mutation.

    Existing approved/rejected teacher decisions are preserved even if their
    descriptions appear contradictory. No polarity or relation type is changed.
    """
    usable, review_items = [], []
    for relation in relations:
        record = copy.deepcopy(relation)
        if (not isinstance(record, dict)
                or record.get("review_status") in ("approved", "rejected")):
            usable.append(record)
            continue
        type_conflict = _explicit_type_conflict(record)
        if type_conflict is not None:
            review_items.append({"reason": "relation_explanation_type_conflict", "rule": _TYPE_CHOICE_RULE_VERSION,
                                 "review_status": "pending", "review_required": True,
                                 "record": record, **type_conflict})
            continue
        negation = _explicit_negation(record) if str(record.get("relation", "")).strip().upper() == "REQUIRES" else None
        if negation is not None:
            review_items.append({"reason": "negated_runtime_requirement", "rule": _NEGATION_RULE_VERSION,
                                 "review_status": "pending", "review_required": True,
                                 "record": record, **negation})
        else:
            usable.append(record)
    return usable, review_items
