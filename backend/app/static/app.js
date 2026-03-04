function $(id) {
  return document.getElementById(id);
}

function toJsonPretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function optionalInputValue(id) {
  const el = $(id);
  if (!el) return undefined;
  const v = (el.value || "").trim();
  return v ? v : undefined;
}

function setDefaultDates() {
  const today = new Date();
  const iso = today.toISOString().slice(0, 10);
  $("asOf").value = iso;

  // Weekly backtest defaults: last ~8 weeks
  const end = new Date(today);
  const start = new Date(today);
  start.setDate(start.getDate() - 56);
  if ($("wbtStart")) $("wbtStart").value = start.toISOString().slice(0, 10);
  if ($("wbtEnd")) $("wbtEnd").value = end.toISOString().slice(0, 10);

  // Monthly backtest defaults (legacy)
  if ($("btEnd")) {
    const ym = today.toISOString().slice(0, 7);
    $("btEnd").value = ym;
    const mstart = new Date(today);
    mstart.setMonth(mstart.getMonth() - 2);
    $("btStart").value = mstart.toISOString().slice(0, 7);
  }
}

async function postJson(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await resp.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    data = { raw: text };
  }
  if (!resp.ok) {
    // FastAPI validation error detail can be list/object
    const msg = typeof data?.detail === "string" ? data.detail : toJsonPretty(data);
    throw new Error(msg);
  }
  return data;
}

function showTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.remove("active");
  });
  document.querySelectorAll(".tab-content").forEach(content => {
    content.classList.remove("active");
  });
  document.querySelector(`[data-tab="${tabName}"]`).classList.add("active");
  $(tabName).classList.add("active");
}

function renderAgentResult(data) {
  $("agentEmpty").classList.add("hidden");
  $("agentResult").classList.remove("hidden");

  // Timeline
  const timelineList = $("timelineList");
  timelineList.innerHTML = "";
  (data.steps || []).forEach(step => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="step-title">${step.name}</div>
      <div class="step-body">${step.output}</div>
    `;
    timelineList.appendChild(li);
  });

  // Portfolio table
  const tbody = document.querySelector("#portfolioTable tbody");
  tbody.innerHTML = "";
  (data.holdings || []).forEach(h => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${h.code}</td>
      <td>${h.name || "-"}</td>
      <td>${(h.weight * 100).toFixed(2)}%</td>
    `;
    tbody.appendChild(tr);
  });

  // Full report
  $("reportPre").textContent = data.rationale || "";
}

function renderBacktestResult(data) {
  $("backtestEmpty").classList.add("hidden");
  $("backtestResult").classList.remove("hidden");

  const { metrics } = data;
  $("cumulativeReturn").textContent = metrics.cumulative_return != null ? `${(metrics.cumulative_return * 100).toFixed(2)}%` : "-";
  $("annualizedReturn").textContent = metrics.annualized_return != null ? `${(metrics.annualized_return * 100).toFixed(2)}%` : "-";
  $("sharpe").textContent = metrics.sharpe != null ? metrics.sharpe.toFixed(3) : "-";
  $("maxDrawdown").textContent = metrics.max_drawdown != null ? `${(metrics.max_drawdown * 100).toFixed(2)}%` : "-";

  drawSingleLineChart($("navChart"), data.nav_dates || [], data.nav_series || [], {
    lineColor: "#7c5cff",
    label: "NAV",
  });
}

function renderWeeklyBacktestResult(data) {
  $("weeklyBacktestEmpty").classList.add("hidden");
  $("weeklyBacktestResult").classList.remove("hidden");

  const ai = data.metrics?.ai || {};
  const bench = data.metrics?.benchmark || {};
  const ex = data.metrics?.excess || {};

  const pct = v => (v != null ? `${(v * 100).toFixed(2)}%` : "-");

  $("aiCumulative").textContent = pct(ai.cumulative_return);
  $("benchCumulative").textContent = pct(bench.cumulative_return);
  $("excessCumulative").textContent = pct(ex.excess_cumulative);
  $("infoRatio").textContent = ex.information_ratio != null ? ex.information_ratio.toFixed(3) : "-";

  $("aiAnnual").textContent = pct(ai.annualized_return);
  $("aiMaxDD").textContent = pct(ai.max_drawdown);
  $("benchAnnual").textContent = pct(bench.annualized_return);
  $("benchMaxDD").textContent = pct(bench.max_drawdown);

  drawDualLineChart(
    $("weeklyNavChart"),
    data.ai_nav_dates || [],
    data.ai_nav_series || [],
    data.bench_nav_dates || [],
    data.bench_nav_series || [],
    {
      aLabel: "AI策略",
      aColor: "#7c5cff",
      bLabel: "基准ETF",
      bColor: "#23c483",
    }
  );
}

function drawSingleLineChart(canvas, dates, series, opts) {
  drawDualLineChart(canvas, dates, series, [], [], {
    aLabel: opts?.label || "Series",
    aColor: opts?.lineColor || "#7c5cff",
    bLabel: "",
    bColor: "rgba(0,0,0,0)",
    hideB: true,
  });
}

function drawDualLineChart(canvas, aDates, aSeries, bDates, bSeries, opts) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width;
  canvas.height = height;

  ctx.clearRect(0, 0, width, height);
  if (!aSeries || aSeries.length === 0) {
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "14px ui-sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("暂无净值数据", width / 2, height / 2);
    return;
  }

  // Align on A dates; if B not same length, sample by date map
  const bMap = new Map();
  (bDates || []).forEach((d, i) => bMap.set(d, (bSeries || [])[i]));
  const bAligned = (aDates || []).map(d => bMap.get(d));

  const allVals = [...aSeries, ...(opts?.hideB ? [] : bAligned.filter(v => v != null))];
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const range = maxVal - minVal || 1;

  const padding = { top: 24, right: 20, bottom: 44, left: 60 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  // Axes
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  // Y labels
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.font = "12px ui-sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartH * i) / 5;
    const v = maxVal - (range * i) / 5;
    ctx.fillText(v.toFixed(3), padding.left - 6, y + 4);
  }

  // X labels (sample at most 8)
  ctx.textAlign = "center";
  const step = Math.ceil((aDates.length || 1) / 8);
  for (let i = 0; i < aDates.length; i += step) {
    const x = padding.left + (chartW * i) / (aDates.length - 1);
    const txt = aDates[i];
    ctx.fillText(txt, x, height - padding.bottom + 16);
  }

  function xAt(i) {
    return padding.left + (chartW * i) / (aSeries.length - 1);
  }

  function yAt(v) {
    return padding.top + chartH * (1 - (v - minVal) / range);
  }

  // Line A
  ctx.strokeStyle = opts?.aColor || "#7c5cff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  aSeries.forEach((v, i) => {
    const x = xAt(i);
    const y = yAt(v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Line B
  if (!opts?.hideB && bAligned && bAligned.some(v => v != null)) {
    ctx.strokeStyle = opts?.bColor || "#23c483";
    ctx.lineWidth = 2;
    ctx.beginPath();
    bAligned.forEach((v, i) => {
      if (v == null) return;
      const x = xAt(i);
      const y = yAt(v);
      // start path on first available
      if (i === 0 || (bAligned[i - 1] == null)) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }

  // Legend
  ctx.font = "12px ui-sans-serif";
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(255,255,255,0.8)";
  ctx.fillText(`${opts?.aLabel || "A"}`, padding.left, 16);
  if (!opts?.hideB) {
    ctx.fillText(`${opts?.bLabel || "B"}`, padding.left + 90, 16);
  }
}

async function runAgent() {
  $("agentStatus").textContent = "运行中...";
  try {
    const body = {
      target_date: $("asOf").value,
      top_k: Number($("topK").value),
      min_holdings: Number($("minHoldings").value),
      model_tag: $("modelTag").value,
      provider: optionalInputValue("provider"),
      model_name: optionalInputValue("modelName"),
    };
    const data = await postJson("/api/agent/run", body);
    renderAgentResult(data);
    $("agentStatus").textContent = "完成";
  } catch (e) {
    $("agentStatus").textContent = "失败";
    alert(e.message);
  }
}

async function runBacktest() {
  $("btStatus").textContent = "运行中...";
  try {
    const body = {
      start_month: $("btStart").value,
      end_month: $("btEnd").value,
      model_tag: $("btModelTag").value,
      provider: optionalInputValue("provider"),
      model_name: optionalInputValue("modelName"),
    };
    const data = await postJson("/api/backtest/run", body);
    renderBacktestResult(data);
    $("btStatus").textContent = "完成";
  } catch (e) {
    $("btStatus").textContent = "失败";
    alert(e.message);
  }
}

async function runWeeklyBacktest() {
  $("wbtStatus").textContent = "运行中...";
  try {
    const body = {
      start_date: $("wbtStart").value,
      end_date: $("wbtEnd").value,
      model_tag: $("wbtModelTag").value,
      provider: optionalInputValue("provider"),
      model_name: optionalInputValue("modelName"),
      benchmark: $("wbtBenchmark").value || "sh.510300",
    };
    const data = await postJson("/api/backtest/weekly", body);
    renderWeeklyBacktestResult(data);
    $("wbtStatus").textContent = "完成";
  } catch (e) {
    $("wbtStatus").textContent = "失败";
    alert(e.message);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  setDefaultDates();
  $("runAgentBtn").addEventListener("click", runAgent);
  if ($("runBacktestBtn")) $("runBacktestBtn").addEventListener("click", runBacktest);
  if ($("runWeeklyBacktestBtn")) $("runWeeklyBacktestBtn").addEventListener("click", runWeeklyBacktest);

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      showTab(btn.dataset.tab);
    });
  });
});
