const DATA = JSON.parse(document.getElementById("page-data").textContent);
const ORDER = ["hybrid", "graphsage", "raw_eng", "rf"];

function series(source, label) {
  return ORDER.map((key) => ({
    key,
    label: `${DATA.model_names[key]} (${label} ${source[key].area.toFixed(3)})`,
    color: token(`--m-${key}`),
    x: source[key].x,
    y: source[key].y,
  }));
}

function draw() {
  lineChart(document.getElementById("roc"), {
    series: series(DATA.roc, "AUC"), xDomain: [0, 1], yDomain: [0, 1], diagonal: true,
    xLabel: "false positive rate", yLabel: "true positive rate", title: "ROC curves on the test period",
  });
  lineChart(document.getElementById("pr"), {
    series: series(DATA.pr, "PR-AUC"), xDomain: [0, 1], yDomain: [0, 1],
    xLabel: "recall", yLabel: "precision", title: "Precision-recall curves on the test period",
  });
  const steps = DATA.per_step.steps;
  lineChart(document.getElementById("per-step"), {
    series: ORDER.map((key) => ({ key, label: DATA.model_names[key], color: token(`--m-${key}`), x: steps, y: DATA.per_step.f1[key] })),
    xDomain: [steps[0] - 0.5, steps[steps.length - 1] + 0.5], yDomain: [0, 1], markers: true, snap: true,
    xTicks: steps.length, xFormat: "d", marker: { x: 42.5, label: "dark-market shutdown" },
    xLabel: "time step", yLabel: "illicit F1", title: "Illicit F1 by test time step", height: 320,
  });
  barChart(document.getElementById("fn"), {
    rows: ORDER.map((key) => ({ label: DATA.model_names[key], value: DATA.fn[key].own, color: token(`--m-${key}`), display: `${DATA.fn[key].own} missed` })),
    xLabel: "false negatives at each model's own threshold", title: "Missed illicit transactions by model",
  });
  barChart(document.getElementById("ablation"), {
    rows: DATA.ablation.map((a) => ({ label: a.label, value: a.test_pr_auc, color: token("--m-graphsage"), display: a.test_pr_auc.toFixed(3) })),
    xDomain: [0, 1], xLabel: "test PR-AUC, GraphSAGE with one seed", title: "Imbalance ablation", labelWidth: 210,
  });
  const f1 = DATA.contrast.filter((c) => c.metric === "f1");
  pairedBars(document.getElementById("contrast"), {
    series: [{ label: "temporal split", color: token("--protocol-temporal") }, { label: "random split", color: token("--protocol-random") }],
    groups: ORDER.map((key) => {
      const row = f1.find((c) => c.model === key);
      return { label: DATA.model_names[key], values: [row.temporal, row.random] };
    }),
    xLabel: "illicit F1", title: "Illicit F1 under the temporal and random protocols",
  });
}

draw();
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(draw, 150);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
