(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.CourseGraphModel = factory();
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const HIERARCHY = new Set(["HAS_PART", "PART_OF", "IS_A"]);
  const MANAGEMENT = new Set(["HAS_EVIDENCE", "CONTAINS", "BELONGS_TO", "NEXT_PAGE", "PREV_PAGE", "PREVIOUS_PAGE",
    "COURSE_CONTAINS_DOCUMENT", "DOCUMENT_CONTAINS_PAGE", "PAGE_CONTAINS_CONTENT"]);
  const ROLES = new Set(["learning", "detail", "context_dependent", "pending"]);
  const STATUS = new Set(["approved", "rejected", "pending"]);
  const text = value => value == null ? "" : String(value);
  const list = value => Array.isArray(value) ? value : [];
  const status = value => STATUS.has(value) ? value : "pending";
  const role = value => ROLES.has(value) ? value : "pending";
  const unique = values => Array.from(new Set(values.filter(Boolean)));
  const props = value => Object.assign({}, value && value.properties || {}, value || {});
  const nodeKey = value => value && typeof value === "object" ? text(value.id == null ? value.name : value.id) : text(value);
  const compare = (a, b) => text(a).localeCompare(text(b), "zh-CN", {numeric: true});

  function finiteData(value) {
    if (typeof value === "number") return Number.isFinite(value);
    if (Array.isArray(value)) return value.every(finiteData);
    if (value && typeof value === "object") return Object.keys(value).every(k => finiteData(value[k]));
    return true;
  }

  function observations(record, warnings, name) {
    let result = record.page_observations;
    if (record.page_observations_json !== undefined) {
      try { result = JSON.parse(record.page_observations_json); }
      catch (_) { result = null; }
    }
    if (result === undefined) return [];
    if (!Array.isArray(result) || result.some(item => !item || typeof item !== "object" || Array.isArray(item)) || !finiteData(result)) {
      warnings.push("概念“" + name + "”的逐页语境格式异常，未作为有效语境使用。");
      return [];
    }
    return result.slice();
  }

  function kindOf(record, nativeConcept) {
    if (nativeConcept) return "concept";
    const labels = list(record.labels).map(label => text(label).toLowerCase());
    const type = text(record.type || record.node_type).toLowerCase();
    const matches = names => names.some(name => labels.includes(name) || type === name);
    if (matches(["concept", "knowledge"])) return "concept";
    if (matches(["course"])) return "course";
    if (matches(["document", "chapter"])) return "document";
    if (matches(["page"])) return "page";
    return "material";
  }

  function sourcePdfLink(raw, source) {
    if (!raw || !source || typeof source !== "object") return null;
    const ref = props(source), pageId = text(ref.page_id), suppliedDocument = text(ref.document_id);
    const stored = list(raw.nodes).map(props);
    const pages = list(raw.pages).map(props).concat(stored.filter(record => kindOf(record, false) === "page"));
    const documents = list(raw.documents).map(props).concat(stored.filter(record => kindOf(record, false) === "document"));
    const matchingPages = pageId ? pages.filter(page => text(page.page_id || page.id) === pageId) : [];
    const documentIds = unique([suppliedDocument].concat(matchingPages.map(page => text(page.document_id))));
    if (documentIds.length !== 1) return null;
    const documentId = documentIds[0];
    const matchingDocuments = documents.filter(document => text(document.document_id || document.id) === documentId);
    const knownPageNumbers = unique(matchingPages.map(page => page.page_no).filter(value => Number.isInteger(value) && value > 0));
    const pageNo = ref.page_no == null ? knownPageNumbers.length === 1 ? knownPageNumbers[0] : null : Number(ref.page_no);
    if (!Number.isInteger(pageNo) || pageNo < 1 || knownPageNumbers.some(value => value !== pageNo)) return null;
    const jobPattern = /^job_[a-f0-9]{32}$/;
    const records = matchingPages.concat(matchingDocuments);
    if (records.some(record => record.job_id && !jobPattern.test(text(record.job_id)))) return null;
    const jobs = unique(records.map(record => text(record.job_id)));
    if (jobs.length > 1) return null;
    let jobId = jobs[0];
    if (!jobId) {
      const integration = props(raw.integration);
      const declaredDocuments = unique([text(integration.document_id)].concat(list(integration.document_ids).map(text)));
      const graphDocuments = unique(documents.map(document => text(document.document_id || document.id)));
      // A latest-job fallback is valid only for an explicitly matched document,
      // or a graph containing exactly that one document. Multi-document graphs
      // must carry their own Page/Document job identities.
      const documentMatches = declaredDocuments.length ? declaredDocuments.includes(documentId) : graphDocuments.length === 1 && graphDocuments[0] === documentId;
      const url = text(integration.source_url), match = /^\/api\/jobs\/(job_[a-f0-9]{32})\/source(?:#page=[1-9][0-9]*)?$/.exec(url);
      if (!documentMatches || !match || integration.job_id && integration.job_id !== match[1]) return null;
      jobId = match[1];
    }
    return {href: "/api/jobs/" + jobId + "/source#page=" + pageNo, pageNo, documentId, jobId};
  }

  function normalize(raw) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new TypeError("图谱必须为包含节点与关系的JSON对象。");
    const native = Array.isArray(raw.concepts);
    if (!native && !Array.isArray(raw.nodes)) throw new TypeError("图谱需要 concepts/relationships 或 nodes/links。");
    const warnings = [], nodes = [], byId = new Map(), byName = new Map();
    const documents = list(raw.documents).map(doc => Object.assign({}, props(doc)));
    const docById = new Map(documents.map(doc => [text(doc.document_id || doc.id), doc]));
    const pages = new Map(list(raw.pages).map(page => {
      const p = props(page);
      return [text(p.page_id || p.id), p];
    }));
    const originals = native ? raw.concepts : raw.nodes;
    for (const original of originals) {
      if (!original || typeof original !== "object" || Array.isArray(original)) { warnings.push("忽略了格式无效的节点。"); continue; }
      const record = props(original), name = text(record.canonical_name || record.name || record.title || record.id).trim();
      const id = text(record.id == null ? record.node_id || (native ? "concept:" + name : name) : record.id).trim();
      if (!id || !name || byId.has(id)) { warnings.push("忽略了名称或ID为空、或ID重复的节点：" + (name || id)); continue; }
      const kind = kindOf(record, native);
      // Frontend layer means material/knowledge layer, not importance or depth.
      const candidateImportance = record.importance_level !== undefined ? record.importance_level : native ? record.layer : undefined;
      const importance = Number(candidateImportance);
      const node = {
        id, name, kind, description: text(record.description),
        aliases: unique(list(record.aliases).map(text)),
        importanceLevel: [1, 2, 3].includes(importance) ? importance : 3,
        teachingRole: role(record.teaching_role), suggestedTeachingRole: role(record.suggested_teaching_role),
        roleReviewStatus: status(record.role_review_status), reviewStatus: status(record.review_status),
        sources: list(record.sources).map(source => Object.assign({}, source)),
        pageObservations: observations(record, warnings, name), raw: original
      };
      nodes.push(node); byId.set(id, node);
      if (!byName.has(name)) byName.set(name, []);
      byName.get(name).push(id);
      if (kind === "document") {
        const did = text(record.document_id || id);
        if (!docById.has(did)) { const doc = Object.assign({}, record, {document_id: did, file_name: record.file_name || name}); documents.push(doc); docById.set(did, doc); }
      } else if (kind === "page") pages.set(text(record.page_id || id), record);
    }
    const resolve = endpoint => {
      const key = nodeKey(endpoint);
      if (byId.has(key)) return key;
      const matches = byName.get(key) || [];
      return matches.length === 1 ? matches[0] : null;
    };
    const links = [], hierarchyCandidates = [], linkIds = new Set();
    const normalizeEdge = (original, index, candidate) => {
      if (!original || typeof original !== "object" || Array.isArray(original)) { warnings.push("忽略了格式无效的关系。"); return; }
      const record = props(original), source = resolve(record.source), target = resolve(record.target);
      if (!source || !target || source === target) { warnings.push("关系端点缺失、歧义或自连，未显示：" + nodeKey(record.source) + " → " + nodeKey(record.target)); return; }
      const relation = text(record.relation || record.relationship_type || record.type || record.value || "RELATED_TO").toUpperCase();
      let id = text(record.id || record.relation_id || record.relationship_id || record.neo4j_element_id || [source, relation, target, index].join("|"));
      if (linkIds.has(id)) id += "|" + index + (candidate ? "|candidate" : "|link");
      linkIds.add(id);
      const edge = {id, source, target, relation, description: text(record.description),
        reviewStatus: status(record.review_status), autoGenerated: record.auto_generated === true, raw: original};
      if (candidate || edge.autoGenerated && (HIERARCHY.has(relation) || relation === "HAS_PARAMETER") && edge.reviewStatus === "pending") {
        hierarchyCandidates.push(edge);
      } else links.push(edge);
    };
    list(native ? raw.relationships : raw.links).forEach((edge, i) => normalizeEdge(edge, i, false));
    list(raw.hierarchy_candidates).forEach((edge, i) => normalizeEdge(edge, i, true));
    const evidenceParents = new Map();
    for (const edge of links) {
      if (edge.relation !== "HAS_EVIDENCE") continue;
      if (!evidenceParents.has(edge.source)) evidenceParents.set(edge.source, []);
      evidenceParents.get(edge.source).push(edge.target);
    }
    for (const node of nodes) {
      if (node.kind !== "concept") continue;
      const record = props(node.raw);
      const evidenceIds = unique(list(record.evidence_node_ids).map(text).concat(evidenceParents.get(node.id) || []));
      const evidence = evidenceIds.map(id => byId.get(id)).filter(Boolean);
      const pageIds = unique(list(record.source_page_ids).map(text).concat(evidence.map(item => {
        const p = props(item.raw);
        return item.kind === "page" ? text(p.page_id || item.id) : text(p.source_page_id || p.page_id);
      })));
      const sources = new Map(node.sources.map(source => [text(source.page_id), source]));
      for (const pid of pageIds) {
        if (sources.has(pid)) continue;
        const page = pages.get(pid) || {};
        sources.set(pid, {
          page_id: pid, document_id: text(page.document_id || record.chapter_id), page_no: page.page_no == null ? null : page.page_no,
          title: text(page.title), extracted_text: text(page.raw_text),
          material_evidence: evidence.filter(item => item.kind === "material" && text(props(item.raw).source_page_id || props(item.raw).page_id) === pid).map(item => ({
            evidence_id: item.id, node_type: text(props(item.raw).node_type || props(item.raw).type || "Material"),
            text: text(props(item.raw).description || props(item.raw).raw_text), review_status: "pending",
            ...Object.fromEntries(["description_scope", "source_kind", "semantic_review_flags", "task_id", "fresh_run", "image_sha256", "same_run_reuse"]
              .filter(key => props(item.raw)[key] !== undefined).map(key => [key, props(item.raw)[key]]))
          }))
        });
      }
      if (!sources.size && record.chapter_id) sources.set("", {page_id: "", document_id: text(record.chapter_id), page_no: null});
      node.sources = Array.from(sources.values());
    }
    return {nodes, links, documents, reviewItems: list(raw.review_items).slice(), hierarchyCandidates,
      warnings, original: raw};
  }

  function roleInfo(node) {
    const labels = {learning: "知识点", detail: "素材细节", context_dependent: "多页用途不一", pending: "待判断"};
    if (node.roleReviewStatus === "approved") return {key: role(node.teachingRole), label: "教师确认 · " + labels[role(node.teachingRole)], confirmed: true};
    if (node.roleReviewStatus === "rejected") return {key: "pending", label: "角色建议未采纳 · 待判断", confirmed: false};
    const key = role(node.suggestedTeachingRole);
    return {key, label: key === "pending" ? "待教师分类" : "模型建议 · " + labels[key], confirmed: false};
  }

  function displayTier(node) {
    // Visual detail is independent of taxonomy, prerequisite order and review.
    // In particular, importance 1 does not turn a material detail into a root.
    const important = Number(node && node.importanceLevel) === 1;
    return roleInfo(node || {}).key === "detail" ? important ? 3 : 4 : important ? 1 : 2;
  }

  function zoomTier(relativeZoom, currentTier) {
    const boundaries = [1.25, 1.8, 2.6], supplied = Number(currentTier);
    const hasCurrent = Number.isInteger(supplied) && supplied >= 1 && supplied <= 4;
    const zoom = Number(relativeZoom);
    if (!Number.isFinite(zoom) || zoom <= 0) return hasCurrent ? supplied : 1;
    if (!hasCurrent) return 1 + boundaries.filter(boundary => zoom >= boundary).length;
    let tier = supplied;
    // Separate entry/exit thresholds prevent wheel jitter from flickering nodes.
    while (tier < 4 && zoom >= boundaries[tier - 1] * 1.08) tier++;
    while (tier > 1 && zoom < boundaries[tier - 2] * 0.92) tier--;
    return tier;
  }

  function conceptNodes(view) {
    return list(view && view.nodes).filter(node => node.kind === "concept" && node.reviewStatus !== "rejected");
  }

  function conceptLinks(view, ids) {
    return list(view && view.links).filter(edge => ids.has(edge.source) && ids.has(edge.target) &&
      edge.reviewStatus !== "rejected" && !MANAGEMENT.has(edge.relation));
  }

  function tierProfile(view) {
    const byId = Object.create(null), counts = [0, 0, 0, 0], cumulative = [0, 0, 0, 0], linkCounts = [0, 0, 0, 0];
    const nodes = conceptNodes(view), ids = new Set(nodes.map(node => node.id));
    for (const node of nodes) { const tier = displayTier(node); byId[node.id] = tier; counts[tier - 1]++; }
    for (let index = 0; index < 4; index++) cumulative[index] = counts[index] + (index ? cumulative[index - 1] : 0);
    for (const edge of conceptLinks(view, ids)) {
      const firstVisible = Math.max(byId[edge.source], byId[edge.target]);
      for (let index = firstVisible - 1; index < 4; index++) linkCounts[index]++;
    }
    const first = counts.findIndex(count => count > 0);
    return {byId, counts, cumulative, linkCounts, total: nodes.length, minTier: first < 0 ? null : first + 1};
  }

  function semanticView(view, tier, selectedId) {
    const requested = Number(tier), current = Number.isInteger(requested) && requested >= 1 && requested <= 4 ? requested : 1;
    const selected = selectedId == null ? "" : text(selectedId), candidates = conceptNodes(view), pinnedIds = [];
    const nodes = candidates.filter(node => {
      if (view.searchActive || displayTier(node) <= current) return true;
      // Pin just this selection from the already filtered view. Do not pull in
      // its neighbours, ancestors, another document or another course.
      if (selected && text(node.id) === selected) { pinnedIds.push(node.id); return true; }
      return false;
    });
    const ids = new Set(nodes.map(node => node.id)), links = conceptLinks(view, ids);
    return Object.assign({}, view, {nodes, links, displayTier: current, hiddenCount: candidates.length - nodes.length, pinnedIds});
  }

  function displayPair(edge) {
    return ["IS_A", "PART_OF"].includes(edge.relation) ? [edge.target, edge.source] : [edge.source, edge.target];
  }

  function primarySource(node) {
    if (node.viewPrimarySource) return node.viewPrimarySource;
    const record = props(node.raw);
    const preferred = node.sources.find(source => record.chapter_id && text(source.document_id) === text(record.chapter_id)) || node.sources[0];
    if (!preferred) return {document_id: text(record.chapter_id), page_no: null};
    return node.sources.filter(source => text(source.document_id) === text(preferred.document_id)).slice()
      .sort((a, b) => pageNumber(a) - pageNumber(b))[0];
  }
  function pageNumber(source) {
    const value = Number(source.page_no);
    return Number.isFinite(value) && value > 0 ? value : Number.POSITIVE_INFINITY;
  }

  function selectView(graph, supplied) {
    const options = Object.assign({mode: "overview", search: "", showDetails: false, showPending: true, documentId: "", importanceLimit: 3}, supplied || {});
    const mode = ["overview", "hierarchy", "prerequisites"].includes(options.mode) ? options.mode : "overview";
    const search = text(options.search).trim().toLocaleLowerCase("zh-CN");
    const limit = [1, 2, 3].includes(Number(options.importanceLimit)) ? Number(options.importanceLimit) : 3;
    const allowedRelation = edge => edge.reviewStatus !== "rejected" &&
      (options.showPending || edge.reviewStatus === "approved") &&
      (mode === "overview" ? !MANAGEMENT.has(edge.relation) :
        mode === "prerequisites" ? edge.relation === "PREREQUISITE_OF" : HIERARCHY.has(edge.relation));
    let hiddenDetails = 0;
    let nodes = graph.nodes.filter(node => {
      if (node.kind !== "concept" || node.reviewStatus === "rejected" || node.importanceLevel > limit) return false;
      if (options.documentId && !node.sources.some(source => text(source.document_id) === text(options.documentId))) return false;
      if (!options.showDetails && !search && roleInfo(node).key === "detail") { hiddenDetails++; return false; }
      return true;
    });
    const searchMode = ["highlight","fade"].includes(options.searchMode) ? options.searchMode : "isolate";
    let matchedIds = null, contextIds = null;
    if (search) {
      const eligible = new Set(nodes.map(node => node.id));
      matchedIds = new Set(nodes.filter(node => [node.name, node.description].concat(node.aliases)
        .join(" ").toLocaleLowerCase("zh-CN").includes(search)).map(node => node.id));
      const retained = new Set(matchedIds);
      // Expand from the original matches only, in either direction; never recurse.
      for (const edge of graph.links) {
        if (!allowedRelation(edge) || !eligible.has(edge.source) || !eligible.has(edge.target)) continue;
        if (matchedIds.has(edge.source)) retained.add(edge.target);
        if (matchedIds.has(edge.target)) retained.add(edge.source);
      }
      contextIds = retained;
      if (searchMode === "isolate") nodes = nodes.filter(node => retained.has(node.id));
    }
    if (options.documentId) nodes = nodes.map(node => Object.assign({}, node, {
      viewPrimarySource: node.sources.filter(source => text(source.document_id) === text(options.documentId)).slice()
        .sort((a, b) => pageNumber(a) - pageNumber(b))[0]
    }));
    const ids = new Set(nodes.map(node => node.id));
    let links = graph.links.filter(edge => {
      if (!ids.has(edge.source) || !ids.has(edge.target) || !allowedRelation(edge)) return false;
      return !matchedIds || searchMode !== "isolate" || matchedIds.has(edge.source) || matchedIds.has(edge.target);
    }).map(edge => {
      const pair = mode === "hierarchy" ? displayPair(edge) : [edge.source, edge.target];
      return Object.assign({}, edge, {displaySource: pair[0], displayTarget: pair[1]});
    });
    const warnings = [];
    if (mode !== "overview") {
      const connected = new Set(links.flatMap(edge => [edge.source, edge.target]));
      nodes = nodes.filter(node => connected.has(node.id));
      if (!links.length) warnings.push(mode === "prerequisites" ? "当前筛选下没有前置关系；请教师结合教学依据补充并确认。" : "当前筛选下没有概念包含或分类关系；来源归组不等于知识层级。");
    }
    return {nodes, links, warnings, hiddenDetails, mode, searchActive: !!search, searchMode, searchMatchIds: [...(matchedIds || [])], searchContextIds: [...(contextIds || [])], documents: graph.documents};
  }

  function layout(view, supplied) {
    const mode = supplied && supplied.mode || view.mode || "overview";
    const positions = Object.create(null), groups = [], components = [], warnings = [];
    const nodeWidth = 240, nodeHeight = 72, gapX = 286, gapY = 106;
    let width = 960, height = 500;
    const grid = list => {
      const byDoc = new Map();
      const documents = new Map((view.documents || []).map(doc => [text(doc.document_id || doc.id), doc]));
      const documentOrder = new Map(Array.from(documents.keys(), (id, index) => [id, index]));
      for (const node of list) {
        const source = primarySource(node), did = text(source.document_id);
        if (!byDoc.has(did)) byDoc.set(did, []);
        byDoc.get(did).push(node);
      }
      let top = 76;
      const groupsBySource = Array.from(byDoc).sort(([a], [b]) =>
        (documentOrder.has(a) ? documentOrder.get(a) : Infinity) - (documentOrder.has(b) ? documentOrder.get(b) : Infinity) || compare(a, b));
      for (const [did, items] of groupsBySource) {
        items.sort((a, b) => pageNumber(primarySource(a)) - pageNumber(primarySource(b)) || a.importanceLevel - b.importanceLevel || compare(a.name, b.name) || compare(a.id, b.id));
        const groupHeight = 48 + Math.ceil(items.length / 3) * gapY;
        const doc = documents.get(did) || {};
        groups.push({key: did, documentId: did, label: text(doc.file_name || doc.chapter_title || doc.name || (did ? did : "未归组")),
          x: 24, y: top, width: 890, height: groupHeight, nodeIds: items.map(item => item.id)});
        items.forEach((node, i) => { positions[node.id] = {x: 38 + i % 3 * gapX, y: top + 44 + Math.floor(i / 3) * gapY}; });
        top += groupHeight + 22;
      }
      height = Math.max(height, top + 30);
    };
    if (mode === "overview" || !view.links.length) {
      grid(view.nodes);
      return {positions, groups, components, width, height, cycle: false, warnings, nodeWidth, nodeHeight};
    }
    const byId = new Map(view.nodes.map(node => [node.id, node]));
    const ids = new Set(byId.keys()), next = new Map(), previous = new Map(), neighbours = new Map();
    const compareNode = (a, b) => compare(byId.get(a).name, byId.get(b).name) || compare(a, b);
    for (const id of ids) { next.set(id, new Set()); previous.set(id, new Set()); neighbours.set(id, new Set()); }
    for (const edge of view.links) {
      const source = edge.displaySource || edge.source, target = edge.displayTarget || edge.target;
      if (!ids.has(source) || !ids.has(target) || next.get(source).has(target)) continue;
      next.get(source).add(target); previous.get(target).add(source);
      neighbours.get(source).add(target); neighbours.get(target).add(source);
    }
    // Stable adjacency makes the drawing independent of database record order.
    for (const map of [next, previous, neighbours]) {
      for (const [id, values] of map) map.set(id, Array.from(values).sort(compareNode));
    }
    const unvisited = new Set(ids), blocks = [];
    for (const start of Array.from(ids).sort(compareNode)) {
      if (!unvisited.has(start)) continue;
      const members = [], todo = [start]; unvisited.delete(start);
      while (todo.length) {
        const id = todo.pop(); members.push(id);
        for (const neighbour of neighbours.get(id)) {
          if (unvisited.delete(neighbour)) todo.push(neighbour);
        }
      }
      members.sort(compareNode);
      blocks.push({members});
    }

    function inversionCount(values) {
      if (values.length < 2) return {values, count: 0};
      const middle = Math.floor(values.length / 2), left = inversionCount(values.slice(0, middle)), right = inversionCount(values.slice(middle));
      const sorted = []; let a = 0, b = 0, count = left.count + right.count;
      while (a < left.values.length && b < right.values.length) {
        if (left.values[a] <= right.values[b]) sorted.push(left.values[a++]);
        else { sorted.push(right.values[b++]); count += left.values.length - a; }
      }
      return {values: sorted.concat(left.values.slice(a), right.values.slice(b)), count};
    }

    function ranked(layers) {
      const result = new Map();
      layers.forEach((layer, level) => layer.forEach((id, index) => result.set(id, {level, index, relative: (index + 0.5) / layer.length})));
      return result;
    }

    function orderScore(layers, edges) {
      const rank = ranked(layers), spans = new Map(); let alignment = 0, crossings = 0;
      for (const [source, target] of edges) {
        const from = rank.get(source), to = rank.get(target), key = from.level + ":" + to.level;
        if (!spans.has(key)) spans.set(key, []);
        spans.get(key).push([from.index, to.index]);
        alignment += Math.abs(from.relative - to.relative);
      }
      for (const pairs of spans.values()) {
        // Sorting shared-source edges by target excludes pairs meeting at a node.
        pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        crossings += inversionCount(pairs.map(pair => pair[1])).count;
      }
      return {crossings, alignment};
    }

    const better = (a, b) => a.crossings < b.crossings || a.crossings === b.crossings && a.alignment < b.alignment - 1e-8;
    const insetX = 24, header = 42, bottom = 24, stepX = 330;
    let cycle = false;
    for (const block of blocks) {
      const indegree = new Map(block.members.map(id => [id, previous.get(id).length]));
      const depth = new Map(block.members.map(id => [id, 0]));
      const roots = block.members.filter(id => !indegree.get(id)), queue = roots.slice();
      let processed = 0;
      while (queue.length) {
        const id = queue.shift(); processed++;
        for (const target of next.get(id)) {
          depth.set(target, Math.max(depth.get(target), depth.get(id) + 1));
          indegree.set(target, indegree.get(target) - 1);
          if (!indegree.get(target)) queue.push(target);
        }
      }
      block.cycle = processed !== block.members.length;
      block.rootIds = roots;
      block.label = (roots.length ? roots : block.members).slice(0, 3).map(id => byId.get(id).name).join(" · ");
      block.local = new Map();
      if (block.cycle) {
        cycle = true;
        const columns = Math.min(3, block.members.length);
        block.width = insetX * 2 + nodeWidth + Math.max(0, columns - 1) * gapX;
        block.height = header + nodeHeight + (Math.ceil(block.members.length / columns) - 1) * gapY + bottom;
        block.members.forEach((id, index) => block.local.set(id, {x: insetX + index % columns * gapX, y: header + Math.floor(index / columns) * gapY}));
        block.levelCount = null; block.crossings = null;
        continue;
      }

      let layers = [];
      for (const id of block.members) {
        const level = depth.get(id);
        if (!layers[level]) layers[level] = [];
        layers[level].push(id);
      }
      for (const layer of layers) layer.sort(compareNode);
      const edges = block.members.flatMap(source => next.get(source).map(target => [source, target]));
      let bestLayers = layers.map(layer => layer.slice()), bestScore = orderScore(layers, edges);
      const remember = () => {
        const score = orderScore(layers, edges);
        if (better(score, bestScore)) { bestScore = score; bestLayers = layers.map(layer => layer.slice()); }
      };
      const barycentric = (level, adjacent) => {
        const rank = ranked(layers), scores = new Map();
        for (const id of layers[level]) {
          let total = 0, weight = 0;
          for (const neighbour of adjacent.get(id)) {
            const entry = rank.get(neighbour), importance = 1 / Math.max(1, Math.abs(level - entry.level));
            total += entry.relative * importance; weight += importance;
          }
          scores.set(id, weight ? total / weight : rank.get(id).relative);
        }
        layers[level].sort((a, b) => scores.get(a) - scores.get(b) || rank.get(a).index - rank.get(b).index || compareNode(a, b));
      };
      for (let sweep = 0; sweep < 6; sweep++) {
        for (let level = 1; level < layers.length; level++) barycentric(level, previous);
        remember();
        for (let level = layers.length - 2; level >= 0; level--) barycentric(level, next);
        remember();
      }
      layers = bestLayers.map(layer => layer.slice());
      // Small local swaps remove ties left by barycentres. Bound this refinement
      // for large graphs; the O(E log E) scoring still applies to normal sweeps.
      if (block.members.length <= 160 && edges.length <= 350) {
        for (let pass = 0; pass < 2; pass++) {
          let improved = false;
          for (const layer of layers) {
            for (let index = 1; index < layer.length; index++) {
              [layer[index - 1], layer[index]] = [layer[index], layer[index - 1]];
              const score = orderScore(layers, edges);
              if (better(score, bestScore)) { bestScore = score; improved = true; }
              else [layer[index - 1], layer[index]] = [layer[index], layer[index - 1]];
            }
          }
          if (!improved) break;
        }
      }
      const rows = Math.max(...layers.map(layer => layer.length)), bodyHeight = nodeHeight + (rows - 1) * gapY;
      const centres = new Map();
      for (const layer of layers) {
        const first = (bodyHeight - (layer.length - 1) * gapY) / 2;
        layer.forEach((id, index) => centres.set(id, first + index * gapY));
      }
      const align = (layer, adjacent) => {
        const wanted = layer.map(id => {
          const values = adjacent.get(id).map(neighbour => centres.get(neighbour));
          return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : centres.get(id);
        });
        // Forward/backward separation keeps the selected order and guarantees
        // enough room for every node even when many parents share one child.
        for (let index = 0; index < wanted.length; index++) wanted[index] = Math.max(nodeHeight / 2 + index * gapY, wanted[index], index ? wanted[index - 1] + gapY : 0);
        for (let index = wanted.length - 1; index >= 0; index--) wanted[index] = Math.min(bodyHeight - nodeHeight / 2 - (wanted.length - 1 - index) * gapY, wanted[index], index < wanted.length - 1 ? wanted[index + 1] - gapY : bodyHeight);
        layer.forEach((id, index) => centres.set(id, wanted[index]));
      };
      for (let pass = 0; pass < 3; pass++) {
        for (let level = 1; level < layers.length; level++) align(layers[level], previous);
        for (let level = layers.length - 2; level >= 0; level--) align(layers[level], next);
      }
      layers.forEach((layer, level) => layer.forEach(id => block.local.set(id, {x: insetX + level * stepX, y: header + centres.get(id) - nodeHeight / 2})));
      block.width = insetX * 2 + nodeWidth + (layers.length - 1) * stepX;
      block.height = header + bodyHeight + bottom;
      block.levelCount = layers.length; block.crossings = bestScore.crossings;
    }
    if (cycle) warnings.push("部分关系存在循环，对应分组暂用网格展示；不能据此生成该组的知识层级或学习顺序。");

    // Pack independent components into separate rectangles instead of putting
    // unrelated branches in one very tall global column. Lowest-bottom fitting
    // uses the space beside a tall component before adding another full row.
    blocks.sort((a, b) => b.width * b.height - a.width * a.height || b.height - a.height || compare(a.members[0], b.members[0]));
    const padding = 24, separation = 36, top = 76;
    const area = blocks.reduce((sum, block) => sum + block.width * block.height, 0);
    const targetWidth = Math.ceil(Math.max(width, ...blocks.map(block => block.width + padding * 2), Math.sqrt(area * 1.7) + padding * 2));
    const placed = [];
    let maximumBottom = top, maximumRight = padding;
    const overlaps = (a, b) => a.x < b.x + b.width + separation && a.x + a.width + separation > b.x && a.y < b.y + b.height + separation && a.y + a.height + separation > b.y;
    for (const block of blocks) {
      const xs = Array.from(new Set([padding, ...placed.map(item => item.x + item.width + separation)])).sort((a, b) => a - b);
      const ys = Array.from(new Set([top, ...placed.map(item => item.y + item.height + separation)])).sort((a, b) => a - b);
      let best = null;
      for (const y of ys) {
        for (const x of xs) {
          const candidate = {x, y, width: block.width, height: block.height};
          if (x + block.width > targetWidth - padding || placed.some(other => overlaps(candidate, other))) continue;
          const low = Math.max(maximumBottom, y + block.height);
          if (!best || low < best.low || low === best.low && (y < best.y || y === best.y && x < best.x)) best = Object.assign({low}, candidate);
        }
      }
      // The lowest existing bottom is always a valid fallback shelf.
      if (!best) best = {x: padding, y: maximumBottom + separation, width: block.width, height: block.height};
      placed.push(best);
      maximumBottom = Math.max(maximumBottom, best.y + block.height);
      maximumRight = Math.max(maximumRight, best.x + block.width);
      for (const [id, point] of block.local) positions[id] = {x: best.x + point.x, y: best.y + point.y};
      const bounds = {x: best.x, y: best.y, width: block.width, height: block.height};
      components.push(Object.assign({key: "component-" + components.length, label: block.label, nodeIds: block.members.slice(), rootIds: block.rootIds.slice(),
        cycle: block.cycle, levelCount: block.levelCount, crossings: block.crossings, bounds}, bounds));
    }
    width = Math.max(width, maximumRight + padding); height = Math.max(height, maximumBottom + padding);
    return {positions, groups, components, width, height, cycle, warnings, nodeWidth, nodeHeight};
  }

  function approvedStudyPath(graph, targetId) {
    const target = text(targetId), ids = new Set(graph.nodes.filter(node => node.kind === "concept" && node.reviewStatus !== "rejected").map(node => node.id));
    if (!ids.has(target)) return {status: "unknown", order: []};
    const parents = new Map(Array.from(ids, id => [id, new Set()]));
    for (const edge of graph.links) if (edge.relation === "PREREQUISITE_OF" && edge.reviewStatus === "approved" && ids.has(edge.source) && ids.has(edge.target)) parents.get(edge.target).add(edge.source);
    const visiting = new Set(), done = new Set(), order = [];
    function visit(id) {
      if (visiting.has(id)) return false;
      if (done.has(id)) return true;
      visiting.add(id);
      for (const parent of Array.from(parents.get(id)).sort(compare)) if (!visit(parent)) return false;
      visiting.delete(id); done.add(id);
      if (id !== target) order.push(id);
      return true;
    }
    return visit(target) ? {status: "ok", order} : {status: "cycle", order: []};
  }

  function wouldCreatePrerequisiteCycle(graph, sourceId, targetId, excludeEdgeId) {
    const source = text(sourceId), target = text(targetId);
    if (source === target) return true;
    const valid = new Set(graph.nodes.filter(node => node.kind === "concept" && node.reviewStatus !== "rejected").map(node => node.id));
    if (!valid.has(source) || !valid.has(target)) return false;
    const next = new Map(Array.from(valid, id => [id, new Set()]));
    for (const edge of graph.links) if (edge.id !== excludeEdgeId && edge.relation === "PREREQUISITE_OF" && edge.reviewStatus !== "rejected" && valid.has(edge.source) && valid.has(edge.target)) next.get(edge.source).add(edge.target);
    const todo = [target], seen = new Set();
    while (todo.length) {
      const id = todo.pop(); if (id === source) return true;
      if (seen.has(id)) continue; seen.add(id);
      todo.push.apply(todo, Array.from(next.get(id)));
    }
    return false;
  }

  function emptyPresentation(input) {
    const value=input || {}, phase=value.phase || "idle";
    if(phase==="loading")return value.hasNodes ? null : {kind:"loading",title:"正在加载课程图谱",description:"正在从数据库读取知识点与关系，请稍候。",busy:true};
    if(phase==="error")return Number(value.status)===401 ?
      {kind:"login",title:"请先登录后查看课程图谱",description:"当前登录状态已失效或尚未登录。登录后可从课程列表重新进入。",action:"login"} :
      {kind:"error",title:"课程图谱加载失败",description:value.error || "未能读取课程图谱，请重新加载。",action:"retry"};
    if(phase!=="ready")return {kind:"idle",title:value.courseSelected ? "尚未载入课程图谱" : "尚未选择课程",description:"可从课程列表进入，或导入已有图谱 JSON。",action:value.courseSelected ? "retry" : "courses"};
    if(value.hasNodes)return null;
    if(!value.conceptCount)return {kind:"empty",title:"课程还没有知识点",description:"此课程尚未发布可展示的知识点，可从课程页处理课件或导入已有图谱。",action:"courses"};
    if(value.mode==="prerequisites")return {kind:"prerequisites",title:"还没有可展示的先修关系",description:"这不代表没有前置知识。请结合教学依据补充、确认先修关系；已有关系也可能被当前筛选收起。",action:"overview"};
    if(value.mode==="hierarchy")return {kind:"hierarchy",title:"当前筛选下没有概念层级",description:"可以展开素材细节或调整筛选。自动归组候选需要单独审核。",action:"overview"};
    return {kind:"filtered",title:"没有符合筛选的知识点",description:"课程已有知识点，请清空搜索或调整课件、重要程度筛选。",action:"clear"};
  }

  return {normalize, roleInfo, displayTier, zoomTier, tierProfile, semanticView, selectView, layout, approvedStudyPath, wouldCreatePrerequisiteCycle, sourcePdfLink, emptyPresentation};
}));
