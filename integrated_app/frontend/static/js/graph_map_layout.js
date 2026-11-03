(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.CourseGraphMapLayout = factory();
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const SIZES = [{width: 240, height: 76, fontSize: 18}, {width: 204, height: 64, fontSize: 16},
    {width: 164, height: 52, fontSize: 14}, {width: 132, height: 42, fontSize: 12}];
  const compare = (a, b) => a < b ? -1 : a > b ? 1 : 0;
  const key = value => value && typeof value === "object" ? String(value.id) : String(value);
  const intersects = (a, b) => a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;

  function rectangle(value) {
    if (!value) return null;
    const box = {id: value.id, x: Number(value.x), y: Number(value.y), width: Number(value.width), height: Number(value.height)};
    return [box.x, box.y, box.width, box.height].every(Number.isFinite) && box.width > 0 && box.height > 0 ? box : null;
  }

  function boxGrid(cellSize) {
    const size = Number(cellSize) > 0 && Number.isFinite(Number(cellSize)) ? Number(cellSize) : 320;
    const cells = new Map();
    const range = box => [Math.floor(box.x / size), Math.floor(box.y / size),
      Math.floor((box.x + box.width) / size), Math.floor((box.y + box.height) / size)];
    function insert(box) {
      const [left, top, right, bottom] = range(box);
      for (let row = top; row <= bottom; row++) for (let column = left; column <= right; column++) {
        const address = column + "," + row;
        if (!cells.has(address)) cells.set(address, []);
        cells.get(address).push(box);
      }
    }
    function any(box) {
      const [left, top, right, bottom] = range(box);
      for (let row = top; row <= bottom; row++) for (let column = left; column <= right; column++) {
        const bucket = cells.get(column + "," + row);
        if (bucket) for (const other of bucket) if (intersects(box, other)) return true;
      }
      return false;
    }
    function query(value) {
      const box = rectangle(value);
      if (!box) return [];
      const [left, top, right, bottom] = range(box), seen = new Set(), result = [];
      for (let row = top; row <= bottom; row++) for (let column = left; column <= right; column++) {
        const bucket = cells.get(column + "," + row);
        if (bucket) for (const other of bucket) {
          if (!seen.has(other.id) && intersects(box, other)) { seen.add(other.id); result.push(other.id); }
        }
      }
      return result;
    }
    return {insert, any, query};
  }

  function spatialIndex(items, cellSize) {
    const grid = boxGrid(cellSize), ids = new Set();
    for (const item of Array.isArray(items) ? items : []) {
      const box = rectangle(item);
      if (box && !ids.has(box.id)) { ids.add(box.id); grid.insert(box); }
    }
    return {query: grid.query};
  }

  function spiral(index) {
    if (!index) return {x: 0, y: 0};
    const ring = Math.ceil((Math.sqrt(index + 1) - 1) / 2), side = ring * 2;
    const remaining = (side + 1) * (side + 1) - 1 - index;
    if (remaining < side) return {x: ring - remaining, y: -ring};
    if (remaining < side * 2) return {x: -ring, y: -ring + remaining - side};
    if (remaining < side * 3) return {x: -ring + remaining - side * 2, y: ring};
    return {x: ring, y: ring - remaining + side * 3};
  }

  function layout(view, supplied) {
    const options = supplied || {}, nodes = Array.isArray(view && view.nodes) ? view.nodes : [];
    const records = new Map(), tiers = new Map(), adjacency = new Map();
    const positions = Object.create(null), nodeSizes = Object.create(null);
    let minimumTier = 4;
    for (const node of nodes) {
      if (!node || node.id == null) continue;
      const id = key(node.id);
      if (records.has(id)) continue;
      const suggested = Number(typeof options.tierOf === "function" ? options.tierOf(node) : node.displayTier || 1);
      const tier = Number.isInteger(suggested) && suggested >= 1 && suggested <= 4 ? suggested : 4;
      records.set(id, node); tiers.set(id, tier); adjacency.set(id, new Set());
      nodeSizes[id] = Object.assign({}, SIZES[tier - 1]); minimumTier = Math.min(minimumTier, tier);
    }
    if (!records.size) return {positions, nodeSizes, width: 960, height: 500, groups: [], components: [], anchorIds: []};
    for (const edge of Array.isArray(view && view.links) ? view.links : []) {
      if (!edge || edge.source == null || edge.target == null) continue;
      const source = key(edge.source), target = key(edge.target);
      if (source !== target && records.has(source) && records.has(target)) {
        adjacency.get(source).add(target); adjacency.get(target).add(source);
      }
    }
    const ids = Array.from(records.keys()).sort(compare), neighbours = new Map(ids.map(id => [id, Array.from(adjacency.get(id)).sort(compare)]));
    const ranked = (a, b) => neighbours.get(b).length - neighbours.get(a).length || compare(a, b);
    const visited = new Set(), parts = [];
    for (const id of ids) {
      if (visited.has(id)) continue;
      const members = [id]; visited.add(id);
      for (let index = 0; index < members.length; index++) for (const next of neighbours.get(members[index])) {
        if (!visited.has(next)) { visited.add(next); members.push(next); }
      }
      members.sort(compare);
      let anchors = members.filter(member => tiers.get(member) === minimumTier).sort(ranked);
      if (!anchors.length) {
        // A disconnected detail-only component still needs a position. Its
        // local anchor is a layout seed, never a new semantic parent/root.
        let localTier = 4;
        for (const member of members) localTier = Math.min(localTier, tiers.get(member));
        anchors = [members.filter(member => tiers.get(member) === localTier).sort(ranked)[0]];
      }
      parts.push({members, anchors});
    }
    parts.sort((a, b) => b.members.length - a.members.length || ranked(a.anchors[0], b.anchors[0]));
    const anchorIds = parts.flatMap(part => part.anchors), anchorSet = new Set(anchorIds);
    const columns = Math.max(1, Math.ceil(Math.sqrt(anchorIds.length * 1.5))), collision = boxGrid(320);
    function place(id, x, y) {
      const size = nodeSizes[id]; positions[id] = {x, y};
      collision.insert({id, x, y, width: size.width, height: size.height});
    }
    // Reserve every core rectangle before adding details. Collisions never
    // push a core anchor, so toggling visible tiers cannot move the map.
    anchorIds.forEach((id, index) => place(id, 24 + index % columns * 320, 24 + Math.floor(index / columns) * 200));
    const owner = new Map(anchorIds.map(id => [id, id])), queue = anchorIds.slice();
    for (let index = 0; index < queue.length; index++) for (const next of neighbours.get(queue[index])) {
      if (!owner.has(next)) { owner.set(next, owner.get(queue[index])); queue.push(next); }
    }
    const assigned = new Map(anchorIds.map(id => [id, []]));
    for (const id of ids) if (!anchorSet.has(id)) assigned.get(owner.get(id)).push(id);
    for (const anchor of anchorIds) {
      const point = positions[anchor], anchorSize = nodeSizes[anchor], centre = {x: point.x + anchorSize.width / 2, y: point.y + anchorSize.height / 2};
      const cursors = [0, 0, 0, 0, 0], members = assigned.get(anchor).sort((a, b) => tiers.get(a) - tiers.get(b) || compare(a, b));
      for (const id of members) {
        const tier = tiers.get(id), size = nodeSizes[id], ring = Math.max(1, tier - tiers.get(anchor));
        let cursor = Math.max(cursors[tier], (ring * 2 - 1) ** 2), candidate;
        // The cursor is monotonic per seed/tier: a dense 5,000-node branch
        // does not restart its empty-space search at the centre for each node.
        do {
          const offset = spiral(cursor++);
          candidate = {id, x: centre.x + offset.x * (size.width + 24) - size.width / 2,
            y: centre.y + offset.y * (size.height + 24) - size.height / 2, width: size.width, height: size.height};
        } while (collision.any({x: candidate.x - 16, y: candidate.y - 16, width: size.width + 32, height: size.height + 32}));
        cursors[tier] = cursor; place(id, candidate.x, candidate.y);
      }
    }
    let minX = Infinity, minY = Infinity, right = -Infinity, bottom = -Infinity;
    for (const id of ids) {
      const p = positions[id], size = nodeSizes[id];
      minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
      right = Math.max(right, p.x + size.width); bottom = Math.max(bottom, p.y + size.height);
    }
    const dx = 24 - minX, dy = 24 - minY;
    for (const id of ids) { positions[id].x += dx; positions[id].y += dy; }
    const components = parts.map((part, index) => {
      let x = Infinity, y = Infinity, r = -Infinity, b = -Infinity;
      for (const id of part.members) {
        const p = positions[id], size = nodeSizes[id];
        x = Math.min(x, p.x); y = Math.min(y, p.y); r = Math.max(r, p.x + size.width); b = Math.max(b, p.y + size.height);
      }
      const bounds = {x, y, width: r - x, height: b - y};
      return Object.assign({key: "map-component-" + index, nodeIds: part.members.slice(), anchorIds: part.anchors.slice(), bounds}, bounds);
    });
    return {positions, nodeSizes, width: Math.max(960, right - minX + 48), height: Math.max(500, bottom - minY + 48), groups: [], components, anchorIds};
  }

  return {layout, spatialIndex};
}));
