function $(id) {
  return document.getElementById(id);
}

function toJsonPretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function getISOWeekString(date) {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const dayNum = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  const weekNo = Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
  return `${d.getUTCFullYear()}-W${String(weekNo).padStart(2, "0")}`;
}

function weekToDate(weekStr) {
  const match = /^(\d{4})-W(\d{2})$/.exec(weekStr || "");
  if (!match) return undefined;
  const year = Number(match[1]);
  const week = Number(match[2]);
  const simple = new Date(Date.UTC(year, 0, 1 + (week - 1) * 7));
  const dayOfWeek = simple.getUTCDay();
  const isoWeekStart = new Date(simple);
  const diff = dayOfWeek <= 4 ? 1 - dayOfWeek : 8 - dayOfWeek;
  isoWeekStart.setUTCDate(simple.getUTCDate() + diff);
  return isoWeekStart.toISOString().slice(0, 10);
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
  if ($("asOf")) $("asOf").value = iso;
  if ($("asOfWeek")) {
    const week = getISOWeekString(today);
    $("asOfWeek").value = week;
  }

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

  const aSeries = data.ai_nav_series || [];
  const aDates = data.ai_nav_dates || [];
  const bSeries = data.bench_nav_series || [];
  const bDates = data.bench_nav_dates || [];

  drawDualLineChart(
    $("weeklyNavChart"),
    aDates,
    aSeries,
    bDates,
    bSeries,
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
    const weekVal = $("asOfWeek") ? $("asOfWeek").value : ( $("asOf") ? $("asOf").value : "" );
    const weekDate = weekToDate(weekVal) || ( $("asOf") ? $("asOf").value : "" ) || new Date().toISOString().slice(0, 10);
    const body = {
      target_date: weekDate,
      top_k: Number($("topK").value),
      min_holdings: 10,
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

function resolveWeeklyModelConfig() {
  const provider = optionalInputValue("wbtProvider") || optionalInputValue("provider");
  const modelName = optionalInputValue("wbtModelName") || optionalInputValue("modelName");
  return { provider, modelName };
}

function setWeeklyProgressHint(text) {
  const el = $("wbtProgressHint");
  if (el) el.textContent = text || "";
}

async function runWeeklyBacktest() {
  $("wbtStatus").textContent = "运行中...";
  setWeeklyProgressHint("正在逐周回测，请稍候（通常在10~60秒，取决于周数与模型响应速度）");
  try {
    const startDate = $("wbtStart").value;
    const endDate = $("wbtEnd").value;
    if (!startDate || !endDate) {
      throw new Error("请先选择开始/结束日期");
    }
    if (startDate > endDate) {
      throw new Error("开始日期不能晚于结束日期");
    }

    const { provider, modelName } = resolveWeeklyModelConfig();
    const alignMode = $("wbtAlignMode")?.value === "true";
    const fixedPortfolio = $("wbtFixedPortfolio")?.value === "true";
    const body = {
      start_date: startDate,
      end_date: endDate,
      model_tag: $("wbtModelTag").value,
      provider,
      model_name: modelName,
      benchmark: $("wbtBenchmark").value || "sh.510300",
      align_mode: alignMode,
      use_fixed_portfolio: fixedPortfolio,
      price_mode: $("wbtPriceMode")?.value || "open_close",
    };
    const data = await postJson("/api/backtest/weekly", body);
    renderWeeklyBacktestResult(data);
    const summary = `完成（${data.processed_weeks || 0}/${data.total_weeks || 0}周）`;
    $("wbtStatus").textContent = summary;
    setWeeklyProgressHint(`已完成：成功 ${data.processed_weeks || 0} 周，失败 ${data.failed_weeks || 0} 周`);
    if (data.warnings && data.warnings.length) {
      alert(`周回测提示：\n- ${data.warnings.slice(0, 8).join("\n- ")}`);
    }
  } catch (e) {
    $("wbtStatus").textContent = "失败";
    setWeeklyProgressHint("运行失败，请根据报错检查日期范围、模型配置或数据源可用性");
    alert(e.message);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  setDefaultDates();
  $("runAgentBtn").addEventListener("click", runAgent);
  if ($("runBacktestBtn")) $("runBacktestBtn").addEventListener("click", runBacktest);
  if ($("runWeeklyBacktestBtn")) $("runWeeklyBacktestBtn").addEventListener("click", runWeeklyBacktest);

  if ($("wbtProvider") && $("provider")) {
    $("wbtProvider").value = "";
  }
  if ($("wbtModelName") && $("modelName")) {
    $("wbtModelName").value = "";
  }
  if ($("wbtPriceMode")) {
    $("wbtPriceMode").value = "open_close";
  }
  if ($("wbtAlignMode")) {
    $("wbtAlignMode").value = "false";
  }
  if ($("wbtFixedPortfolio")) {
    $("wbtFixedPortfolio").value = "false";
  }

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      showTab(btn.dataset.tab);
    });
  });
});
