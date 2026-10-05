const fieldsEl = document.getElementById("fields");
const form = document.getElementById("predictForm");
const btn = document.getElementById("predictBtn");
const btnText = document.getElementById("btnText");
const result = document.getElementById("result");

let fieldNames = [];
const nodeEls = {};   // tree node id -> <g>
const edgeEls = {};   // child node id -> <path> (edge coming into that node)

async function init() {
  try {
    const res = await fetch("/api/meta");
    if (!res.ok) throw new Error("Server error");
    const meta = await res.json();

    buildForm(meta.fields);
    document.getElementById("testAcc").textContent = meta.test_accuracy + "%";
    document.getElementById("accNote").textContent =
      `Trained on ${meta.train_size.toLocaleString()} mushrooms, tested on ${meta.test_size.toLocaleString()} unseen ones.`;
    drawAccuracyChart(meta);
    drawPie(meta.distribution);
    document.getElementById("datasetNote").textContent =
      `${(meta.distribution.edible + meta.distribution.poisonous).toLocaleString()} mushrooms: ` +
      `${meta.distribution.edible.toLocaleString()} edible, ${meta.distribution.poisonous.toLocaleString()} poisonous.`;
    drawTree(meta.tree);
  } catch (err) {
    fieldsEl.innerHTML = '<p class="muted">Could not load data. Is the Flask server running?</p>';
  }
}

function buildForm(fields) {
  fieldsEl.innerHTML = "";
  fieldNames = fields.map(f => f.name);
  fields.forEach(f => {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const label = document.createElement("label");
    label.htmlFor = f.name;
    label.textContent = f.title;
    const select = document.createElement("select");
    select.id = f.name;
    f.options.forEach(o => {
      const opt = document.createElement("option");
      opt.value = o.value;
      opt.textContent = o.label;
      select.appendChild(opt);
    });
    wrap.append(label, select);
    fieldsEl.appendChild(wrap);
  });
}

/* ---------- charts ---------- */
function drawAccuracyChart(m) {
  new Chart(document.getElementById("accChart"), {
    type: "bar",
    data: {
      labels: ["Training", "Test"],
      datasets: [{
        data: [m.train_accuracy, m.test_accuracy],
        backgroundColor: ["#b9e0a5", "#6cc487"],
        borderRadius: 8,
      }],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      scales: {
        x: { min: 0, max: 100, ticks: { color: "#f3efe0", callback: v => v + "%" }, grid: { color: "rgba(243,239,224,.12)" } },
        y: { ticks: { color: "#f3efe0" }, grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => c.parsed.x + "%" } },
      },
    },
  });
}

function drawPie(d) {
  new Chart(document.getElementById("pieChart"), {
    type: "pie",
    data: {
      labels: ["Edible", "Poisonous"],
      datasets: [{ data: [d.edible, d.poisonous], backgroundColor: ["#6cc487", "#e0674e"], borderColor: "rgba(10,31,20,.8)", borderWidth: 3 }],
    },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } } },
  });
}

/* ---------- decision tree drawing (plain SVG) ---------- */
function drawTree(root) {
  const NS = "http://www.w3.org/2000/svg";
  const W = 132, H = 46, GAP_X = 16, GAP_Y = 78;
  let leafCount = 0, maxDepth = 0;

  // Layout: leaves get evenly spaced x; parents sit in the middle of their children.
  (function layout(n, depth) {
    n.depth = depth;
    maxDepth = Math.max(maxDepth, depth);
    if (n.leaf) {
      n.x = leafCount++ * (W + GAP_X) + W / 2;
    } else {
      layout(n.no, depth + 1);
      layout(n.yes, depth + 1);
      n.x = (n.no.x + n.yes.x) / 2;
    }
    n.y = depth * (H + GAP_Y) + 4;
  })(root, 0);

  const width = leafCount * (W + GAP_X) - GAP_X;
  const height = maxDepth * (H + GAP_Y) + H + 8;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const edges = document.createElementNS(NS, "g");
  const nodes = document.createElementNS(NS, "g");
  svg.append(edges, nodes);

  function el(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (text !== undefined) e.textContent = text;
    return e;
  }

  (function draw(n) {
    const g = el("g", { class: "node" + (n.leaf ? " leaf-" + n.prediction.toLowerCase() : "") });
    g.append(el("rect", { class: "box", x: n.x - W / 2, y: n.y, width: W, height: H, rx: 12 }));
    const line1 = n.leaf ? n.prediction : n.feature;
    const line2 = n.leaf ? `${n.samples.toLocaleString()} mushrooms` : `is ${n.value}?`;
    g.append(el("text", { class: "t1", x: n.x, y: n.y + 19 }, line1));
    g.append(el("text", { class: "t2", x: n.x, y: n.y + 35 }, line2));
    nodes.appendChild(g);
    nodeEls[n.id] = g;

    if (!n.leaf) {
      [["no", n.no], ["yes", n.yes]].forEach(([name, child]) => {
        const x1 = n.x, y1 = n.y + H, x2 = child.x, y2 = child.y, my = (y1 + y2) / 2;
        const p = el("path", { class: "edge", d: `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}` });
        edges.appendChild(p);
        edgeEls[child.id] = p;
        edges.appendChild(el("text", { class: "edge-label", x: (x1 + x2) / 2 + (name === "no" ? -14 : 14), y: my - 2 }, name === "yes" ? "Yes" : "No"));
        draw(child);
      });
    }
  })(root);

  document.getElementById("tree").appendChild(svg);
}

function highlightPath(path) {
  Object.values(nodeEls).forEach(g => g.classList.remove("on"));
  Object.values(edgeEls).forEach(p => p.classList.remove("on"));
  path.forEach(id => {
    if (nodeEls[id]) nodeEls[id].classList.add("on");
    if (edgeEls[id]) edgeEls[id].classList.add("on");
  });
}

/* ---------- prediction ---------- */
function showResult(state, icon, title, info) {
  result.dataset.state = state;
  document.getElementById("resultIcon").textContent = icon;
  document.getElementById("resultTitle").textContent = title;
  document.getElementById("resultInfo").textContent = info;
  result.classList.remove("pop");
  void result.offsetWidth;
  result.classList.add("pop");
}

// Enter inside a dropdown also submits
form.addEventListener("keydown", e => {
  if (e.key === "Enter" && e.target.tagName === "SELECT") {
    e.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async e => {
  e.preventDefault();
  const payload = {};
  fieldNames.forEach(n => (payload[n] = document.getElementById(n).value));

  btn.disabled = true;
  btnText.textContent = "Predicting…";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");

    const poisonous = data.prediction === "Poisonous";
    showResult(
      poisonous ? "poisonous" : "edible",
      poisonous ? "☠️" : "✅",
      data.prediction,
      `Confidence ${data.confidence}% · model test accuracy ${data.test_accuracy}%`
    );
    highlightPath(data.path);
  } catch (err) {
    showResult("error", "⚠️", "Something went wrong", err.message);
  } finally {
    btn.disabled = false;
    btnText.textContent = "Predict";
  }
});

init();
