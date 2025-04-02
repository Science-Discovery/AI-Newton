import re
from collections import defaultdict


def extract_expression_pattern(expr_list: list[str],
                               equiv_dict: dict[str, set[str]],
                               minimal_terms: int = 1) -> tuple[str, dict[str, set[str]]]:
    """
    从一个表达式列表中提取它们通用的模板和待匹配的模式的已有符号集合。
    """
    if len(expr_list) < minimal_terms:
        return (None, None)

    # 构建符号到通配符的映射
    symbol_to_wildcard = {}
    for wildcard, symbols in equiv_dict.items():
        for symbol in symbols:
            symbol_to_wildcard[symbol] = wildcard

    # 处理每个表达式，生成模板并收集符号
    templates = []
    all_collected = defaultdict(set)
    pattern = re.compile(r'([a-zA-Z_]\w*)(\[\d+\])?')

    for expr in expr_list:
        collected = defaultdict(set)

        def replacer(match):
            root = match.group(1)
            index = match.group(2) or ''
            wildcard = symbol_to_wildcard.get(root, root)
            if wildcard in equiv_dict:
                collected[wildcard].add(root)
            return f"{wildcard}{index}"

        template = pattern.sub(replacer, expr)
        templates.append(template)

        for wc, symbols in collected.items():
            all_collected[wc].update(symbols)

    # 检查所有模板是否相同
    if len(set(templates)) != 1:
        return (None, None)
    common_template = templates[0]

    # 检查收集的符号是否合法
    for wc, symbols in all_collected.items():
        if not symbols.issubset(equiv_dict.get(wc, set())):
            return (None, None)

    # 构建结果字典
    # result = {}
    # for wc in all_collected:
    #     result[wc] = (common_template, all_collected[wc])
    result = (common_template, dict(all_collected))  # (common_template, {pattern: {symbols}})

    return result


def extract_partials(s: str) -> list[tuple[str, str, int]]:
    # pattern = r'Partial\[([^]]+)\]/Partial\[(pos[xyz])\[(\d+)\]\]'
    pattern = r'Partial\[(.*?)\]/Partial\[(pos[xyz])\[(\d+)\]\]'
    return [
        (m[0], m[1], int(m[2]))
        for m in re.findall(pattern, s)
        if int(m[2]) > 0
    ]


def remove_outer_parentheses(s):
    s = s.strip()
    while s.startswith('(') and s.endswith(')'):
        stack = 0
        can_remove = True
        for i, c in enumerate(s):
            if c == '(':
                stack += 1
            elif c == ')':
                stack -= 1
            if stack == 0 and i != len(s) - 1:
                can_remove = False
                break
            if stack < 0:
                can_remove = False
                break
        if can_remove:
            s = s[1:-1].strip()
        else:
            break
    return s


def split_top_level_plus(expr):
    parts = []
    current = []
    depth = 0
    for char in expr:
        if char == '+' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            current.append(char)
    parts.append(''.join(current).strip())
    return parts


def expand_expression(expr):
    expr = remove_outer_parentheses(expr)
    parts = split_top_level_plus(expr)
    if len(parts) > 1:
        terms = []
        for part in parts:
            part_terms = expand_expression(part)
            terms.extend(part_terms)
        return terms
    else:
        return [expr]


def format_expression(expr: str) -> str:
    terms = expand_expression(expr)
    wrapped_terms = [f'({term})' for term in terms]
    return ' + '.join(wrapped_terms)


def generate_pos_dict(expressions):
    original_set = set(expressions)
    template_dict = {}
    pos_pattern = re.compile(r'(/Partial\[)(pos[xyz])(\[\d+\]\])')

    for expr in expressions:
        match = pos_pattern.search(expr)
        if not match:
            continue
        prefix, pos, suffix = match.groups()
        template = pos_pattern.sub(rf'\g<1>pos_\g<3>', expr)
        if template not in template_dict:
            template_dict[template] = set()
        template_dict[template].add(pos)

    result = {"posx": [], "posy": [], "posz": []}
    all_pos = {"posx", "posy", "posz"}

    for template, poses in template_dict.items():
        if len(poses) == 2:
            missing = (all_pos - poses).pop()
            new_expr = template.replace("pos_", missing, 1)
            if new_expr not in original_set:
                result[missing].append(new_expr)

    return result


def count_gradient_components(expressions: list[str]) -> int:
    template_dict = {}
    pos_pattern = re.compile(r'(/Partial\[)(pos[xyz])(\[\d+\]\])')

    for expr in expressions:
        match = pos_pattern.search(expr)
        if not match:
            continue
        prefix, pos, suffix = match.groups()
        template = pos_pattern.sub(rf'\g<1>pos_\g<3>', expr)
        if template not in template_dict:
            template_dict[template] = set()
        template_dict[template].add(pos)

    if all([not poses for poses in template_dict.values()]):
        return 0

    return len(set.union(*[poses for poses in template_dict.values()]))


def check_single_pos(expressions):
    pos_pattern = re.compile(r'pos[xyz]\[\d+\]')
    found_pos = set()

    for expr in expressions:
        matches = pos_pattern.findall(expr)
        found_pos.update(matches)

    unique_pos = {pos.split('[')[0] for pos in found_pos}

    if len(unique_pos) == 1:
        return unique_pos.pop()
    else:
        return False
