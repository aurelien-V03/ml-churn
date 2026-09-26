// Served by the API, the page calls it on the same origin. Opened as a file,
// it targets the local service: otherwise the fetch would look for
// file:///predict and fail without a response.
const API = location.protocol.startsWith("http") ? "" : "http://127.0.0.1:8000";

// Service status, refreshed continuously: one green dot per probe that answers,
// red as soon as it fails or the service is unreachable.
const STATUS_PERIOD_MS = 5000;

// Predictions run on their own as soon as both probes answer, and again after
// the service has been away -- the table would otherwise keep stale results.
let served = false;

// Predicted columns, appended after the client's own data.
const PREDICTED = [
  { key: "churn", label: "churn" },
  { key: "lifetime", label: "estimation vie client" },
];

const head = document.getElementById("head");
const body = document.getElementById("body");
const modelSelect = document.getElementById("model");
const progress = document.getElementById("progress");
const error = document.getElementById("error");

function buildTable() {
  const columns = ["client_id", ...FEATURES];

  for (const name of columns) {
    const cell = document.createElement("th");
    cell.textContent = name;
    head.appendChild(cell);
  }
  for (const { label } of PREDICTED) {
    const cell = document.createElement("th");
    cell.textContent = label;
    cell.className = "predicted";
    head.appendChild(cell);
  }

  CLIENTS.forEach((client, index) => {
    const row = document.createElement("tr");
    for (const name of columns) {
      const cell = document.createElement("td");
      cell.textContent = client[name];
      row.appendChild(cell);
    }
    for (const { key } of PREDICTED) {
      const cell = document.createElement("td");
      cell.className = "predicted";
      cell.id = `${key}-${index}`;
      cell.textContent = "—";
      row.appendChild(cell);
    }
    body.appendChild(row);
  });
}

// One call per client: the endpoint scores a single row at a time.
async function predictOne(client, index) {
  const { client_id, ...data } = client;
  const response = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: modelSelect.value, data }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(`HTTP ${response.status} — ${JSON.stringify(payload.detail)}`);

  const cell = document.getElementById(`churn-${index}`);
  cell.textContent = `${payload.churn ? "oui" : "non"} (${payload.probability})`;
  cell.classList.toggle("churn-true", payload.churn);
  cell.classList.toggle("churn-false", !payload.churn);
}

// Guards the loop: a model change must not overlap a run already going.
let running = false;

async function predictAll() {
  if (running) return;
  running = true;
  error.textContent = "";

  try {
    for (const [index, client] of CLIENTS.entries()) {
      progress.textContent = `${index + 1} / ${CLIENTS.length}`;
      try {
        await predictOne(client, index);
      } catch (failure) {
        error.textContent = `${client.client_id} : ${failure.message}`;
        progress.textContent = `interrompu à ${index + 1} / ${CLIENTS.length}`;
        return;
      }
    }
    progress.textContent = `${CLIENTS.length} clients prédits`;
  } finally {
    running = false;
  }
}

// Changer de modele relance la serie : les colonnes predites viendraient
// sinon de deux modeles differents.
modelSelect.addEventListener("change", predictAll);

async function probe(path) {
  try {
    const response = await fetch(`${API}${path}`, { cache: "no-store" });
    return { ok: response.ok, body: await response.json() };
  } catch {
    return { ok: false, body: null };
  }
}

function paint(dot, ok) {
  dot.classList.toggle("ok", ok);
  dot.classList.toggle("ko", !ok);
}

// The dropdown mirrors the list announced by /ready: a model added on the
// service side shows up here without touching the page.
function syncModels(models) {
  const current = [...modelSelect.options].map((option) => option.value);
  if (current.join() === models.join()) return;

  const chosen = modelSelect.value;
  modelSelect.innerHTML = "";
  for (const name of models) {
    const option = document.createElement("option");
    option.value = option.textContent = name;
    modelSelect.appendChild(option);
  }
  if (models.includes(chosen)) modelSelect.value = chosen;
}

async function refreshStatus() {
  const [health, ready] = await Promise.all([probe("/health"), probe("/ready")]);

  paint(document.getElementById("dot-health"), health.ok);
  paint(document.getElementById("dot-ready"), ready.ok);

  const detail = document.getElementById("status-detail");
  if (!health.ok && !ready.ok) {
    detail.textContent = "service injoignable";
  } else if (ready.ok) {
    detail.textContent = `modèles : ${ready.body.models.join(", ")}`;
    syncModels(ready.body.models);
  } else {
    detail.textContent = "modèles non chargés";
  }

  const pret = health.ok && ready.ok;
  if (pret && !served) predictAll();
  served = pret;
}

buildTable();
refreshStatus();
setInterval(refreshStatus, STATUS_PERIOD_MS);
