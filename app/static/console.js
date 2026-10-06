const DATA = JSON.parse(document.getElementById("page-data").textContent);
const byId = (id) => document.getElementById(id);
const pct = d3.format(".1%");
const signed = d3.format("+.3f");
const ORDER = ["hybrid", "graphsage", "raw_eng", "rf"];
const SHAPES = { illicit: d3.symbolSquare, licit: d3.symbolCircle, unknown: d3.symbolTriangle };
let current = null;

function say(text) {
  byId("message").textContent = text || "";
}

async function getJSON(url, options) {
  const response = await fetch(url, options);
  if (!(response.headers.get("Content-Type") || "").includes("json")) {
    throw new Error(`the server answered with status ${response.status} and no data; a sleeping instance takes about a minute to wake, so try again shortly`);
  }
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `request failed with status ${response.status}`);
  return body;
}

function flag(on) {
  const span = document.createElement("span");
  span.className = "flag";
  const svg = d3.create("svg").attr("viewBox", "0 0 12 12").attr("aria-hidden", "true");
  if (on) svg.append("path").attr("d", "M6 1 L11 11 L1 11 Z").attr("fill", token("--critical"));
  else svg.append("circle").attr("cx", 6).attr("cy", 6).attr("r", 5).attr("fill", token("--good"));
  span.append(svg.node(), document.createTextNode(on ? "Flagged as illicit" : "Not flagged"));
  return span;
}

function facts(meta, result) {
  const list = byId("tx-meta");
  list.replaceChildren();
  const rows = [
    ["Transaction", meta.id === "new" ? "new" : String(meta.id), true],
    ["Time step", String(result.time_step), false],
    ["Dataset label", meta.label, false],
    ["Threshold", pct(result.threshold), false],
  ];
  if (result.latency_ms !== undefined) rows.push(["Server time", `${result.latency_ms} ms`, false]);
  rows.forEach(([term, value, isId]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = value;
    if (isId) dd.className = "id";
    list.append(dt, dd);
  });
}

function graphLegend() {
  const list = document.createElement("ul");
  list.className = "legend";
  [["illicit", "illicit"], ["licit", "licit"], ["unknown", "unlabelled"]].forEach(([key, text]) => {
    const item = document.createElement("li");
    const shape = d3.create("svg").attr("class", "shape").attr("viewBox", "-7 -7 14 14").attr("aria-hidden", "true");
    shape.append("path").attr("d", d3.symbol(SHAPES[key], 70)()).attr("fill", token("--seq-2"));
    item.append(shape.node(), document.createTextNode(text));
    list.append(item);
  });
  const ramp = document.createElement("li");
  const bar = document.createElement("span");
  bar.className = "ramp";
  ramp.append(document.createTextNode("lower risk"), bar, document.createTextNode("higher risk"));
  list.append(ramp);
  const focus = document.createElement("li");
  focus.textContent = "dark outline: the transaction being scored";
  list.append(focus);
  return list;
}

function drawGraph(hood, focus) {
  const box = byId("graph");
  box.replaceChildren(graphLegend());
  const width = Math.max(300, box.clientWidth - 24 || 640);
  const height = 400;
  const nodes = hood.nodes.map((n) => ({ ...n }));
  const links = hood.edges.map(([source, target]) => ({ source, target }));
  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(46))
    .force("charge", d3.forceManyBody().strength(-160))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide(13))
    .stop();
  for (let i = 0; i < 300; i += 1) sim.tick();
  nodes.forEach((d) => {
    d.x = Math.max(16, Math.min(width - 16, d.x));
    d.y = Math.max(16, Math.min(height - 16, d.y));
  });
  const ramp = d3.scaleLinear().domain([0, 0.25, 0.5, 0.75, 1]).range(["--seq-0", "--seq-1", "--seq-2", "--seq-3", "--seq-4"].map(token)).clamp(true);
  const svg = d3.select(box).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img")
    .attr("aria-label", `Neighbourhood of transaction ${focus}: ${nodes.length} transactions within two hops`);
  svg.append("defs").append("marker").attr("id", "arrow").attr("viewBox", "0 0 8 8").attr("refX", 15).attr("refY", 4)
    .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
    .append("path").attr("d", "M0 0 L8 4 L0 8 Z").attr("fill", token("--axis"));
  svg.append("g").selectAll("line").data(links).join("line")
    .attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y)
    .attr("stroke", token("--axis")).attr("stroke-width", 1.2).attr("marker-end", "url(#arrow)");
  svg.append("g").selectAll("path").data(nodes).join("path")
    .attr("transform", (d) => `translate(${d.x},${d.y})`)
    .attr("d", (d) => d3.symbol(SHAPES[d.label], d.id === focus ? 320 : 150)())
    .attr("fill", (d) => ramp(d.risk))
    .attr("stroke", (d) => (d.id === focus ? token("--ink") : token("--page")))
    .attr("stroke-width", (d) => (d.id === focus ? 2.5 : 2))
    .style("cursor", (d) => (d.id === "new" || d.id === focus ? "default" : "pointer"))
    .on("mousemove", (event, d) => tip(event, [d.id === "new" ? "new transaction" : `transaction ${d.id}`, `${d.label}, ${d.hop} hop${d.hop === 1 ? "" : "s"} away`, `risk ${pct(d.risk)}${d.flagged ? ", flagged" : ""}`]))
    .on("mouseleave", hideTip)
    .on("click", (event, d) => {
      if (d.id !== "new" && d.id !== focus) lookup(d.id);
    });
  byId("graph-note").textContent = `${nodes.length} transactions shown${hood.truncated ? ", the nearest 60" : ""}. Select one to score it.`;
}

function drawDrivers(drivers) {
  const box = byId("drivers");
  box.replaceChildren();
  const row = 28;
  const m = { top: 4, right: 60, bottom: 26, left: 170 };
  const height = drivers.length * row + m.top + m.bottom;
  const { svg, width } = frame(box, height, "Contributions of the strongest drivers to the hybrid model's log-odds");
  const max = d3.max(drivers, (d) => Math.abs(d.contribution)) || 1;
  const x = d3.scaleLinear().domain([-max, max]).range([m.left, width - m.right]);
  svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.top).attr("y2", height - m.bottom).attr("stroke", token("--axis"));
  const rows = svg.selectAll("g.row").data(drivers).join("g").attr("transform", (d, i) => `translate(0,${m.top + i * row})`);
  rows.append("text").attr("class", "id").attr("x", m.left - 10).attr("y", row / 2).attr("text-anchor", "end").attr("dominant-baseline", "middle").text((d) => d.feature);
  rows.append("path").attr("d", (d) => barPath(x(0), d.contribution >= 0 ? Math.max(x(0) + 2, x(d.contribution)) : Math.min(x(0) - 2, x(d.contribution)), 6, row - 12))
    .attr("fill", (d) => (d.contribution >= 0 ? token("--toward-illicit") : token("--toward-licit")))
    .on("mousemove", (event, d) => tip(event, [d.feature, `${d.kind === "raw" ? "raw value" : "computed value"} ${d.value === null ? "none (no neighbours in that direction)" : d.value.toPrecision(6)}`, `contribution ${signed(d.contribution)} log-odds`]))
    .on("mouseleave", hideTip);
  rows.append("text").attr("class", "value").attr("x", (d) => (d.contribution >= 0 ? x(d.contribution) + 5 : x(d.contribution) - 5))
    .attr("text-anchor", (d) => (d.contribution >= 0 ? "start" : "end")).attr("y", row / 2).attr("dominant-baseline", "middle").text((d) => signed(d.contribution));
  svg.append("text").attr("class", "note").attr("x", m.left).attr("y", height - 6).text("toward licit");
  svg.append("text").attr("class", "note").attr("x", width - m.right).attr("y", height - 6).attr("text-anchor", "end").text("toward illicit");
  const table = byId("drivers-table");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  ["Feature", "Kind", "Value", "Contribution"].forEach((text, i) => {
    const th = document.createElement("th");
    th.textContent = text;
    if (i > 1) th.className = "num";
    head.append(th);
  });
  const body = table.createTBody();
  drivers.forEach((d) => {
    const tr = body.insertRow();
    [d.feature, d.kind === "raw" ? "raw value" : "computed", d.value === null ? "none" : d.value.toPrecision(6), signed(d.contribution)].forEach((text, i) => {
      const td = tr.insertCell();
      td.textContent = text;
      if (i > 1) td.className = "num";
    });
  });
}

function drawScores(scores) {
  const rows = ORDER.filter((key) => key in scores).map((key) => ({
    label: DATA.model_names[key],
    value: scores[key],
    color: token(`--m-${key}`),
    display: `${pct(scores[key])}, threshold ${pct(DATA.thresholds[key])}`,
  }));
  barChart(byId("scores"), { rows, xDomain: [0, 1], xFormat: ".0%", xLabel: "probability of illicit", title: "Scores from every model", labelWidth: 190 });
}

function prefillNewForm(meta) {
  if (meta.id === "new") return;
  byId("new-step").value = String(meta.step);
  byId("new-inputs").value = String(meta.id);
  byId("new-outputs").value = "";
}

function render(result, meta) {
  byId("result").hidden = false;
  byId("risk-value").textContent = pct(result.risk);
  byId("risk-meter").querySelector(".fill").style.width = `${result.risk * 100}%`;
  byId("risk-meter").querySelector(".tick").style.left = `calc(${result.threshold * 100}% - 1px)`;
  byId("risk-caption").replaceChildren(flag(result.flagged), document.createTextNode(`, threshold ${pct(result.threshold)}`));
  facts(meta, result);
  drawGraph(result.neighbourhood, meta.id);
  drawDrivers(result.drivers);
  drawScores(result.scores);
  prefillNewForm(meta);
  current = { result, meta };
}

async function lookup(id) {
  say("Scoring...");
  try {
    const result = await getJSON(`/api/transactions/${encodeURIComponent(String(id).trim())}`);
    render(result, { id: result.tx_id, step: result.time_step, label: result.label });
    byId("tx-input").value = String(result.tx_id);
    history.replaceState(null, "", `?tx=${result.tx_id}`);
    say("");
  } catch (error) {
    say(error.message);
  }
}

async function sample(label) {
  try {
    const body = await getJSON(`/api/transactions/sample?label=${label}`);
    await lookup(body.tx_id);
  } catch (error) {
    say(error.message);
  }
}

function ids(text) {
  return text.split(/[\s,]+/).filter(Boolean).map((part) => (/^\d+$/.test(part) ? Number(part) : part));
}

async function scoreNew(event) {
  event.preventDefault();
  const features = DATA.raw_names.map((name) => DATA.medians[name]);
  DATA.form_features.forEach((name) => {
    const text = byId(`f-${name}`).value.trim();
    if (text) features[DATA.raw_names.indexOf(name)] = Number(text);
  });
  const body = {
    time_step: Number(byId("new-step").value),
    features,
    inputs: ids(byId("new-inputs").value),
    outputs: ids(byId("new-outputs").value),
  };
  say("Scoring the new transaction...");
  try {
    const result = await getJSON("/api/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    render(result, { id: "new", step: result.time_step, label: "not labelled" });
    say("");
    byId("result").scrollIntoView({ block: "start" });
  } catch (error) {
    say(error.message);
    byId("message").scrollIntoView({ block: "center" });
  }
}

function buildForm() {
  const select = byId("new-step");
  DATA.steps.forEach((step) => select.append(new Option(String(step), String(step))));
  const fields = byId("new-fields");
  DATA.form_features.forEach((name) => {
    const wrap = document.createElement("div");
    const label = document.createElement("label");
    const input = document.createElement("input");
    label.htmlFor = `f-${name}`;
    label.textContent = name;
    input.id = `f-${name}`;
    input.type = "number";
    input.step = "any";
    input.value = String(DATA.medians[name]);
    input.placeholder = String(DATA.medians[name]);
    wrap.append(label, input);
    fields.append(wrap);
  });
}

byId("lookup").addEventListener("submit", (event) => {
  event.preventDefault();
  if (byId("tx-input").value.trim()) lookup(byId("tx-input").value);
});
document.querySelectorAll("button.sample").forEach((button) => button.addEventListener("click", () => sample(button.dataset.label)));
byId("new-form").addEventListener("submit", scoreNew);
buildForm();
const initial = new URLSearchParams(location.search).get("tx");
if (initial) lookup(initial);
else sample("illicit");
let resizeTimer = null;
const redraw = () => {
  if (current) render(current.result, current.meta);
};
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(redraw, 150);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
