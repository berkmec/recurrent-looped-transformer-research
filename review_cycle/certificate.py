"""Exact affine residual certificates. Never substitutes an oracle answer."""
from fractions import Fraction
import hashlib
import heapq
import json
import math


class EvidenceError(ValueError):
    pass


def digest(obj):
    data = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def number(value):
    if type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value)):
        raise EvidenceError('NONFINITE_OR_INVALID_NUMBER')
    return Fraction(value)


def threshold(value):
    number(value)
    result = Fraction(str(value))
    if result < 0:
        raise EvidenceError('NEGATIVE_THRESHOLD')
    return result


def rational(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def name(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EvidenceError('INVALID_NAME')
    return value


def parse_evidence(events):
    """Return latest constraints and a globally consistent affine graph.

    Floating inputs are their exact represented rationals. Unanchored unrelated
    components are permitted; contradictory components are never permitted.
    """
    if not isinstance(events, list) or not events:
        raise EvidenceError('MISSING_EVIDENCE')
    latest = {}
    for event in events:
        if not isinstance(event, dict):
            raise EvidenceError('INVALID_EVENT')
        kind = event.get('kind')
        if kind == 'continuation':
            continue
        if kind == 'query':
            name(event.get('child'))
            continue
        if kind not in ('root', 'edge', 'revision'):
            raise EvidenceError('UNSUPPORTED_EVENT')
        child = name(event.get('child'))
        parent = event.get('parent')
        if parent is not None:
            parent = name(parent)
        if kind == 'root' and parent is not None:
            raise EvidenceError('ROOT_WITH_PARENT')
        if kind == 'edge' and parent is None:
            raise EvidenceError('EDGE_WITHOUT_PARENT')
        value = number(event.get('value'))
        key = ('absolute:' if parent is None else 'edge:') + child
        latest[key] = (child, parent, value)
    graph, anchors = {}, {}
    for key, (child, parent, value) in latest.items():
        graph.setdefault(child, [])
        if parent is None:
            anchors[child] = (value, key)
        else:
            graph.setdefault(parent, [])
            graph[parent].append((child, value, key))
            graph[child].append((parent, -value, key))
    visited = set()
    component_anchors = {}
    for start in sorted(graph):
        if start in visited:
            continue
        potential = {start: Fraction(0)}
        queue = [start]
        for node in queue:
            for other, offset, _ in graph[node]:
                proposed = potential[node] + offset
                if other in potential:
                    if potential[other] != proposed:
                        raise EvidenceError('INCONSISTENT_CYCLE')
                else:
                    potential[other] = proposed
                    queue.append(other)
        origins = {anchors[n][0] - potential[n] for n in queue if n in anchors}
        if len(origins) > 1:
            raise EvidenceError('INCONSISTENT_ANCHORS')
        for node in queue:
            component_anchors[node] = bool(origins)
        visited.update(queue)
    return latest, graph, anchors, component_anchors


def _candidate(candidate, graph):
    if not isinstance(candidate, dict):
        raise EvidenceError('INVALID_CANDIDATE')
    values = {}
    for key, value in candidate.items():
        name(key)
        if key not in graph:
            raise EvidenceError('UNKNOWN_CANDIDATE_NODE')
        values[key] = number(value)
    return values


def _unresolved(reason):
    return dict(status='UNRESOLVED_BUDGET', bound=None, reason=reason,
                checked_edges=0, certificate=None)


def verify_path(candidate, events, query, tolerance=.1):
    try:
        limit = threshold(tolerance)
        name(query)
        _, graph, anchors, anchored = parse_evidence(events)
        values = _candidate(candidate, graph)
        if query not in graph or query not in values:
            raise EvidenceError('MISSING_QUERY_CANDIDATE')
        if not anchored[query]:
            raise EvidenceError('UNANCHORED_QUERY')
        distance, predecessor, heap = {}, {}, []
        for node, (value, key) in sorted(anchors.items()):
            if node in values:
                residual = abs(values[node] - value)
                distance[node] = residual
                predecessor[node] = (None, key, residual)
                heapq.heappush(heap, (residual, node))
        while heap:
            dist, node = heapq.heappop(heap)
            if distance.get(node) != dist:
                continue
            if node == query:
                break
            for other, offset, key in graph[node]:
                if other not in values:
                    continue
                residual = abs(values[other] - values[node] - offset)
                proposed = dist + residual
                if other not in distance or proposed < distance[other]:
                    distance[other] = proposed
                    predecessor[other] = (node, key, residual)
                    heapq.heappush(heap, (proposed, other))
        if query not in distance:
            raise EvidenceError('NO_COMPLETE_CANDIDATE_PATH')
        path, terms, node = [], [], query
        while node is not None:
            path.append(node)
            previous, key, residual = predecessor[node]
            terms.append(dict(facet=key, residual=rational(residual)))
            node = previous
        path.reverse()
        terms.reverse()
        bound = distance[query]
        cert = dict(version=1, query=query, path=path, terms=terms,
                    bound=rational(bound), threshold=rational(limit),
                    source_sha256=digest(events), candidate_sha256=digest(candidate))
        try:
            display_bound = float(bound)
        except OverflowError:
            display_bound = None
        return dict(status='VERIFIED' if bound <= limit else 'CONTINUE',
                    bound=display_bound, bound_exact=rational(bound),
                    display_bound_overflow=display_bound is None,
                    reason='exact_anchor_path_residual_bound', checked_edges=len(terms),
                    candidate_answer=candidate[query], certificate=cert)
    except (EvidenceError, TypeError, KeyError, OverflowError, ValueError) as exc:
        return _unresolved(str(exc))


def audit_certificate(cert, candidate, events, query, tolerance=.1):
    """Recompute supplied path terms without shortest-path search."""
    try:
        limit = threshold(tolerance)
        latest, graph, anchors, anchored = parse_evidence(events)
        values = _candidate(candidate, graph)
        if (cert['version'] != 1 or cert['query'] != query
                or cert['source_sha256'] != digest(events)
                or cert['candidate_sha256'] != digest(candidate)
                or not anchored.get(query)):
            return False
        path = cert['path']
        if not path or path[-1] != query or len(path) != len(set(path)):
            return False
        if len(cert['terms']) != len(path) or path[0] not in anchors:
            return False
        value, facet = anchors[path[0]]
        residual = abs(values[path[0]] - value)
        expected = [dict(facet=facet, residual=rational(residual))]
        total = residual
        for prev, node, term in zip(path, path[1:], cert['terms'][1:]):
            child, parent, offset = latest[term['facet']]
            if child == node and parent == prev:
                signed = offset
            elif child == prev and parent == node:
                signed = -offset
            else:
                return False
            residual = abs(values[node] - values[prev] - signed)
            expected.append(dict(facet=term['facet'], residual=rational(residual)))
            total += residual
        return (cert['terms'] == expected and cert['bound'] == rational(total)
                and cert['threshold'] == rational(limit) and total <= limit)
    except (EvidenceError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return False


def bind_candidate(candidate, events, query, scope):
    name(scope)
    return dict(nodes=dict(candidate), source_sha256=digest(events), query=query, scope=scope)


def verify_envelope(envelope, events, query, scope, tolerance=.1):
    try:
        name(scope)
        if envelope.get('scope') != scope:
            raise EvidenceError('SCOPE_MISMATCH')
        if envelope.get('query') != query:
            raise EvidenceError('QUERY_MISMATCH')
        if envelope.get('source_sha256') != digest(events):
            raise EvidenceError('SOURCE_VERSION_MISMATCH')
        return verify_path(envelope['nodes'], events, query, tolerance)
    except (EvidenceError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return _unresolved(str(exc))
