/* Metadata for newly generated graph files; preserves all original PNG chunks. */
(function (root, factory) {
  "use strict";
  const value = factory();
  if (typeof module === "object" && module.exports) module.exports = value;
  else root.CourseExportMetadata = value;
})(typeof window !== "undefined" ? window : this, function () {
  "use strict";
  const author = "昕煜 冷";
  function crc32(bytes) {
    let crc = 0xffffffff;
    for (const value of bytes) {
      crc ^= value;
      for (let bit = 0; bit < 8; bit++) crc = crc >>> 1 ^ (crc & 1 ? 0xedb88320 : 0);
    }
    return (crc ^ 0xffffffff) >>> 0;
  }
  async function pngWithAuthor(blob) {
    const source = new Uint8Array(await blob.arrayBuffer());
    if (source.length < 20 || ![137,80,78,71,13,10,26,10].every((value, index) => source[index] === value)) throw Error("无法写入图片作者信息：PNG格式不正确");
    const view = new DataView(source.buffer, source.byteOffset, source.byteLength);
    let offset = 8;
    while (offset + 12 <= source.length) {
      const length = view.getUint32(offset);
      if (offset + 12 + length > source.length) throw Error("PNG数据不完整");
      const type = String.fromCharCode(...source.subarray(offset + 4, offset + 8));
      if (type === "IDAT" || type === "IEND") break;
      offset += length + 12;
    }
    if (offset + 12 > source.length) throw Error("PNG缺少结束标记");
    const payload = new TextEncoder().encode("Author\0\0\0\0\0" + author);
    const chunk = new Uint8Array(payload.length + 12), chunkView = new DataView(chunk.buffer);
    chunkView.setUint32(0, payload.length); chunk.set([105,84,88,116], 4); chunk.set(payload, 8);
    chunkView.setUint32(payload.length + 8, crc32(chunk.subarray(4, payload.length + 8)));
    return new Blob([source.subarray(0, offset), chunk, source.subarray(offset)], {type: "image/png"});
  }
  function annotateSvg(svg) {
    const doc = svg.ownerDocument, rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#", dc = "http://purl.org/dc/elements/1.1/";
    const meta = doc.createElementNS("http://www.w3.org/2000/svg", "metadata");
    const graph = doc.createElementNS(rdf, "rdf:RDF"), description = doc.createElementNS(rdf, "rdf:Description");
    const creator = doc.createElementNS(dc, "dc:creator"), sequence = doc.createElementNS(rdf, "rdf:Seq"), item = doc.createElementNS(rdf, "rdf:li");
    item.textContent = author; sequence.append(item); creator.append(sequence); description.append(creator); graph.append(description); meta.append(graph); svg.prepend(meta);
    svg.setAttribute("font-family", "SimSun, 宋体, serif");
    for (const text of svg.querySelectorAll("text")) { text.setAttribute("font-family", "SimSun, 宋体, serif"); text.style.fontFamily = 'SimSun, "宋体", serif'; }
    return svg;
  }
  return {author, pngWithAuthor, annotateSvg};
});
