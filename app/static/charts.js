const tooltip = document.createElement("div");
tooltip.className = "tooltip";
tooltip.setAttribute("role", "tooltip");
document.body.append(tooltip);

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function tip(event, lines) {
  tooltip.textContent = [].concat(lines).join("\n");
  tooltip.style.opacity = "1";
  const pad = 14;
  const box = tooltip.getBoundingClientRect();
  let left = event.clientX + pad;
  let top = event.clientY + pad;
  if (left + box.width > window.innerWidth - 8) left = event.clientX - box.width - pad;
  if (top + box.height > window.innerHeight - 8) top = event.clientY - box.height - pad;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function hideTip() {
  tooltip.style.opacity = "0";
}

function legend(box, series) {
  const list = document.createElement("ul");
  list.className = "legend";
  series.forEach((s) => {
    const item = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = s.color;
    item.append(swatch, document.createTextNode(s.label));
    list.append(item);
  });
  box.append(list);
}

function frame(box, height, label) {
  const width = Math.max(280, box.clientWidth || 560);
  const svg = d3.select(box).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img").attr("aria-label", label);
  return { svg, width };
}

function nearest(values, target) {
  let best = 0;
  values.forEach((v, i) => {
    if (Math.abs(v - target) < Math.abs(values[best] - target)) best = i;
  });
  return best;
}

function barPath(x0, x1, y, h) {
  const r = Math.min(4, Math.abs(x1 - x0), h / 2);
  const dir = x1 >= x0 ? 1 : -1;
  return `M${x0},${y}H${x1 - dir * r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - dir * r},${y + h}H${x0}Z`;
}

function axes(svg, x, y, width, height, m, spec) {
  svg.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line")
    .attr("x1", m.left).attr("x2", width - m.right).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.bottom})`)
    .call(d3.axisBottom(x).ticks(spec.xTicks || 6, spec.xFormat).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
  svg.append("text").attr("x", (m.left + width - m.right) / 2).attr("y", height - 6).attr("text-anchor", "middle").text(spec.xLabel);
  svg.append("text").attr("transform", `translate(12,${(m.top + height - m.bottom) / 2}) rotate(-90)`).attr("text-anchor", "middle").text(spec.yLabel);
}

function lineChart(box, spec) {
  box.replaceChildren();
  legend(box, spec.series);
  const height = spec.height || 300;
  const m = { top: 14, right: 18, bottom: 42, left: 50 };
  const { svg, width } = frame(box, height, spec.title);
  const x = d3.scaleLinear().domain(spec.xDomain).range([m.left, width - m.right]);
  const y = d3.scaleLinear().domain(spec.yDomain).range([height - m.bottom, m.top]);
  axes(svg, x, y, width, height, m, spec);
  if (spec.diagonal) {
    svg.append("line").attr("class", "reference").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1));
  }
  if (spec.marker && spec.marker.x > spec.xDomain[0] && spec.marker.x < spec.xDomain[1]) {
    svg.append("line").attr("class", "reference").attr("x1", x(spec.marker.x)).attr("x2", x(spec.marker.x)).attr("y1", m.top).attr("y2", height - m.bottom);
    svg.append("text").attr("class", "note").attr("x", x(spec.marker.x) + 6).attr("y", m.top + 10).text(spec.marker.label);
  }
  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
  spec.series.forEach((s) => {
    const points = s.x.map((v, i) => [v, s.y[i]]);
    svg.append("path").attr("d", line(points)).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("stroke-linejoin", "round").attr("stroke-linecap", "round");
    if (spec.markers) {
      svg.append("g").selectAll("circle").data(points).join("circle")
        .attr("cx", (d) => x(d[0])).attr("cy", (d) => y(d[1])).attr("r", 4)
        .attr("fill", s.color).attr("stroke", token("--surface")).attr("stroke-width", 2);
    }
  });
  const cross = svg.append("line").attr("class", "cross").attr("y1", m.top).attr("y2", height - m.bottom).attr("opacity", 0);
  svg.append("rect").attr("x", m.left).attr("y", m.top).attr("width", width - m.left - m.right).attr("height", height - m.top - m.bottom).attr("fill", "transparent")
    .on("mousemove", (event) => {
      const value = x.invert(d3.pointer(event)[0]);
      const at = spec.snap ? spec.series[0].x[nearest(spec.series[0].x, value)] : value;
      cross.attr("x1", x(at)).attr("x2", x(at)).attr("opacity", 1);
      const lines = [`${spec.xLabel} ${spec.snap ? at : at.toFixed(2)}`];
      spec.series.forEach((s) => lines.push(`${s.label}: ${s.y[nearest(s.x, at)].toFixed(3)}`));
      tip(event, lines);
    })
    .on("mouseleave", () => {
      cross.attr("opacity", 0);
      hideTip();
    });
}

function barChart(box, spec) {
  box.replaceChildren();
  const row = 30;
  const m = { top: 4, right: 72, bottom: 40, left: spec.labelWidth || 200 };
  const height = spec.rows.length * row + m.top + m.bottom;
  const { svg, width } = frame(box, height, spec.title);
  const x = d3.scaleLinear().domain(spec.xDomain || [0, d3.max(spec.rows, (r) => r.value) || 1]).nice().range([m.left, width - m.right]);
  svg.append("g").attr("class", "grid").selectAll("line").data(x.ticks(5)).join("line")
    .attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", m.top).attr("y2", height - m.bottom);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.bottom})`).call(d3.axisBottom(x).ticks(5, spec.xFormat).tickSizeOuter(0));
  svg.append("text").attr("x", (m.left + width - m.right) / 2).attr("y", height - 6).attr("text-anchor", "middle").text(spec.xLabel);
  const rows = svg.selectAll("g.row").data(spec.rows).join("g").attr("class", "row").attr("transform", (d, i) => `translate(0,${m.top + i * row})`);
  rows.append("text").attr("x", m.left - 10).attr("y", row / 2).attr("text-anchor", "end").attr("dominant-baseline", "middle").text((d) => d.label);
  rows.append("path").attr("d", (d) => barPath(x(0), Math.max(x(0) + 2, x(d.value)), 7, row - 14)).attr("fill", (d) => d.color)
    .on("mousemove", (event, d) => tip(event, [d.label, d.display])).on("mouseleave", hideTip);
  rows.append("text").attr("class", "value").attr("x", (d) => x(d.value) + 6).attr("y", row / 2).attr("dominant-baseline", "middle").text((d) => d.display);
}

function pairedBars(box, spec) {
  box.replaceChildren();
  legend(box, spec.series);
  const band = 46;
  const m = { top: 4, right: 72, bottom: 40, left: spec.labelWidth || 200 };
  const height = spec.groups.length * band + m.top + m.bottom;
  const { svg, width } = frame(box, height, spec.title);
  const x = d3.scaleLinear().domain([0, 1]).range([m.left, width - m.right]);
  svg.append("g").attr("class", "grid").selectAll("line").data(x.ticks(5)).join("line")
    .attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", m.top).attr("y2", height - m.bottom);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.bottom})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  svg.append("text").attr("x", (m.left + width - m.right) / 2).attr("y", height - 6).attr("text-anchor", "middle").text(spec.xLabel);
  spec.groups.forEach((group, i) => {
    const top = m.top + i * band;
    svg.append("text").attr("x", m.left - 10).attr("y", top + band / 2).attr("text-anchor", "end").attr("dominant-baseline", "middle").text(group.label);
    spec.series.forEach((s, j) => {
      const value = group.values[j];
      const y = top + 7 + j * 16;
      svg.append("path").attr("d", barPath(x(0), Math.max(x(0) + 2, x(value)), y, 12)).attr("fill", s.color)
        .on("mousemove", (event) => tip(event, [group.label, `${s.label}: ${value.toFixed(3)}`])).on("mouseleave", hideTip);
      svg.append("text").attr("class", "value").attr("x", x(value) + 6).attr("y", y + 6).attr("dominant-baseline", "middle").text(value.toFixed(3));
    });
  });
}
