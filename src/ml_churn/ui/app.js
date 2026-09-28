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

// Only one model per family so far; the request carries its name anyway.
const MODEL = "xgboost";

// Columns shown in the table. The requests still carry every feature the
// models need: only the display is trimmed.
const DISPLAYED = [
  "client_id",
  "anciennete_mois",
  "sieges_souscrits",
  "revenu_mensuel_recurrent_eur",
  "utilisateurs_actifs",
  "taux_adoption_pct",
  "connexions_30j",
  "heures_usage_30j",
  "derniere_connexion_jours",
];

// Predicted columns, appended after the client's own data.
const PREDICTED = [
  { key: "churn", label: "churn" },
  { key: "lifetime", label: "estimation vie client" },
];

// The key comes from the service, which reads it from the `.env` the page has
// no access to. Fetched once, before the first prediction.
let API_KEY = "";

async function loadApiKey() {
  const response = await fetch(`${API}/ui-config`, { cache: "no-store" });
  API_KEY = (await response.json()).api_key;
}

const head = document.getElementById("head");
const body = document.getElementById("body");
const progress = document.getElementById("progress");
const error = document.getElementById("error");

// Clients shown at once. The predictions, they, cover the whole set.
const PAGE_SIZE = 20;
let page = 0;

// Predictions, kept by client index: the table is redrawn on every sort and
// must be able to restore what has already been predicted.
const PREDICTIONS = new Map();

// Column currently sorted, and its direction. `null` keeps the source order.
let sortedBy = null;
let ascending = true;

function columnValue(index, column) {
  const prediction = PREDICTIONS.get(index);
  // Not predicted yet: pushed to the end whatever the direction.
  if (column === "churn") return prediction?.probability ?? -1;
  if (column === "lifetime") return prediction?.lifetime ?? -1;

  const raw = CLIENTS[index][column];
  return typeof raw === "number" ? raw : String(raw);
}

function sortedOrder() {
  const order = CLIENTS.map((_, index) => index);
  if (sortedBy === null) return order;

  return order.sort((left, right) => {
    const a = columnValue(left, sortedBy);
    const b = columnValue(right, sortedBy);
    const sens = ascending ? 1 : -1;
    if (a < b) return -sens;
    if (a > b) return sens;
    return 0;
  });
}

function buildHead() {
  const columns = [
    ...DISPLAYED.map((name) => ({ key: name, label: name, predicted: false })),
    ...PREDICTED.map(({ key, label }) => ({ key, label, predicted: true })),
  ];

  // Redessine l'en-tete : la fleche du tri change de colonne a chaque clic.
  head.replaceChildren();

  for (const { key, label, predicted } of columns) {
    const sorted = key === sortedBy;
    const cell = document.createElement("th");
    cell.dataset.column = key;
    cell.textContent = sorted ? `${label} ${ascending ? "▲" : "▼"}` : label;
    cell.classList.toggle("predicted", predicted);
    cell.classList.toggle("sorted", sorted);
    // Un clic trie, un second inverse le sens.
    cell.addEventListener("click", () => {
      ascending = sortedBy === key ? !ascending : true;
      sortedBy = key;
      // Trier renvoie au debut : rester page 4 apres un tri n'a aucun sens.
      page = 0;
      buildHead();
      renderRows();
    });
    head.appendChild(cell);
  }
}

function pageCount() {
  return Math.max(1, Math.ceil(CLIENTS.length / PAGE_SIZE));
}

function renderPagination() {
  const premier = page * PAGE_SIZE;
  document.getElementById("page-indicator").textContent =
    `${premier + 1} – ${Math.min(premier + PAGE_SIZE, CLIENTS.length)} sur ${CLIENTS.length}` +
    `  ·  page ${page + 1} / ${pageCount()}`;
  document.getElementById("previous").disabled = page === 0;
  document.getElementById("next").disabled = page >= pageCount() - 1;
}

function goToPage(numero) {
  page = Math.min(Math.max(numero, 0), pageCount() - 1);
  renderRows();
}

document.getElementById("previous").addEventListener("click", () => goToPage(page - 1));
document.getElementById("next").addEventListener("click", () => goToPage(page + 1));

function renderRows() {
  body.replaceChildren();
  renderPagination();

  // Le tri porte sur l'ensemble des clients, la page n'en montre qu'une tranche.
  for (const index of sortedOrder().slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)) {
    const client = CLIENTS[index];
    const row = document.createElement("tr");

    for (const name of DISPLAYED) {
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
    showPrediction(index);
  }
}

function showPrediction(index) {
  const prediction = PREDICTIONS.get(index);
  const churnCell = document.getElementById(`churn-${index}`);
  // Le client peut appartenir a une autre page : sa ligne n'existe pas.
  if (!prediction || !churnCell) return;

  churnCell.textContent = `${prediction.churn ? "oui" : "non"} (${prediction.probability})`;
  churnCell.classList.toggle("churn-true", prediction.churn);
  churnCell.classList.toggle("churn-false", !prediction.churn);

  document.getElementById(`lifetime-${index}`).textContent = EUROS.format(
    prediction.lifetime,
  );
}

// One call per client and per endpoint: each scores a single row at a time.
async function ask(path, data) {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": API_KEY },
    body: JSON.stringify({ model: MODEL, data }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`${path} — HTTP ${response.status} — ${JSON.stringify(payload.detail)}`);
  }
  return payload;
}

const EUROS = new Intl.NumberFormat("fr-FR", {
  style: "currency",
  currency: "EUR",
  maximumFractionDigits: 0,
});

async function predictOne(client, index) {
  const { client_id, ...data } = client;
  const [churn, clv] = await Promise.all([
    ask("/predict-churn", data),
    ask("/predict-clv", data),
  ]);

  PREDICTIONS.set(index, {
    probability: churn.probability,
    churn: churn.churn,
    lifetime: clv.lifetime_value_eur,
  });
  showPrediction(index);
}

// Guards the loop: a restart must not overlap a run already going.
let running = false;

async function predictAll() {
  if (running) return;

  try {
    if (!API_KEY) await loadApiKey();
  } catch (failure) {
    error.textContent = `Clé d'API indisponible : ${failure.message}`;
    return;
  }
  if (!API_KEY) {
    error.textContent = "FAST_API_KEY n'est pas configurée côté service.";
    return;
  }
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



// Drift report: the service compares the training extract to the current one.
const DRIFT_COLUMNS = [
  { key: "column", label: "colonne" },
  { key: "psi", label: "PSI" },
  { key: "ks", label: "KS" },
  { key: "p_value", label: "p-value" },
  { key: "verdict", label: "verdict" },
];

async function loadDrift() {
  // La population courante, ce sont les clients affiches : ils partent au
  // service, qui les compare a l'extrait ayant servi a l'entrainement.
  const data = CLIENTS.map(({ client_id, ...features }) => features);

  const response = await fetch(`${API}/drift`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": API_KEY },
    body: JSON.stringify({ data }),
  });
  if (!response.ok) return;
  const report = await response.json();

  document.getElementById("drift-source").textContent =
    `${report.reference} (référence) comparé à ${report.current} — ` +
    `PSI sous 0.10 : stable, sous 0.25 : dérive modérée, au-delà : importante.`;

  const head = document.getElementById("drift-head");
  head.replaceChildren();
  for (const { label } of DRIFT_COLUMNS) {
    const cell = document.createElement("th");
    cell.textContent = label;
    head.appendChild(cell);
  }

  const body = document.getElementById("drift-body");
  body.replaceChildren();
  for (const ligne of report.columns) {
    const row = document.createElement("tr");
    for (const { key } of DRIFT_COLUMNS) {
      const cell = document.createElement("td");
      cell.textContent = ligne[key];
      if (key === "verdict") cell.className = `verdict-${ligne.verdict}`;
      // Le service tranche sur la p-valeur complete, pas sur celle arrondie.
      if (key === "p_value" && ligne.significant) cell.className = "significatif";
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
}

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

async function refreshStatus() {
  const [health, ready] = await Promise.all([probe("/health"), probe("/ready")]);

  paint(document.getElementById("dot-health"), health.ok);
  paint(document.getElementById("dot-ready"), ready.ok);

  const detail = document.getElementById("status-detail");
  if (!health.ok && !ready.ok) {
    detail.textContent = "service injoignable";
  } else if (ready.ok) {
    // `models` arrive par famille : {"churn": [...], "clv": [...]}.
    const familles = Object.entries(ready.body.models)
      .map(([famille, noms]) => `${famille} : ${noms.join(", ")}`)
      .join("  ·  ");
    detail.textContent = familles;
  } else {
    detail.textContent = "modèles non chargés";
  }

  const pret = health.ok && ready.ok;
  if (pret && !served) predictAll();
  served = pret;
}

buildHead();
renderRows();
refreshStatus();
setInterval(refreshStatus, STATUS_PERIOD_MS);

loadApiKey().then(loadDrift);
