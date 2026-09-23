// Value ranges observed on the training extract, column by column.
const BOUNDS = {
  "anciennete_mois": {
    "min": 1,
    "max": 36,
    "integer": true
  },
  "sieges_souscrits": {
    "min": 1,
    "max": 897,
    "integer": true
  },
  "utilisateurs_actifs": {
    "min": 0,
    "max": 829,
    "integer": true
  },
  "taux_adoption_pct": {
    "min": 0.0,
    "max": 100.0,
    "integer": false
  },
  "connexions_30j": {
    "min": 0,
    "max": 156,
    "integer": true
  },
  "heures_usage_30j": {
    "min": 0.0,
    "max": 136.7,
    "integer": false
  },
  "fonctionnalites_utilisees": {
    "min": 0,
    "max": 40,
    "integer": true
  },
  "nb_integrations": {
    "min": 0,
    "max": 16,
    "integer": true
  },
  "derniere_connexion_jours": {
    "min": 0,
    "max": 200,
    "integer": true
  },
  "tickets_support_90j": {
    "min": 0,
    "max": 14,
    "integer": true
  },
  "delai_reponse_support_h": {
    "min": 0.5,
    "max": 56.5,
    "integer": false
  },
  "csat": {
    "min": 1,
    "max": 5,
    "integer": true
  },
  "retards_paiement_12m": {
    "min": 0,
    "max": 7,
    "integer": true
  },
  "revenu_mensuel_recurrent_eur": {
    "min": 9.11,
    "max": 76511.23,
    "integer": false
  },
  "inactivite_relative": {
    "min": 0.0,
    "max": 1.0,
    "integer": false
  },
  "inactif_30j": {
    "min": 0,
    "max": 1,
    "integer": true
  },
  "taux_fonctionnalites": {
    "min": 0.0,
    "max": 1.0,
    "integer": false
  },
  "polarite_csm_alerte": {
    "min": 0,
    "max": 1,
    "integer": true
  },
  "polarite_csm_neutre": {
    "min": 0,
    "max": 1,
    "integer": true
  },
  "polarite_csm_positif": {
    "min": 0,
    "max": 1,
    "integer": true
  },
  "polarite_csm_absent": {
    "min": 0,
    "max": 1,
    "integer": true
  },
  "niveau_anciennete_recent": {
    "min": 0,
    "max": 1,
    "integer": true
  },
  "niveau_anciennete_etabli": {
    "min": 0,
    "max": 1,
    "integer": true
  },
  "niveau_anciennete_ancien": {
    "min": 0,
    "max": 1,
    "integer": true
  }
};

// One-hot columns: exactly one modality of each group is set to 1.
const GROUPS = {
  "polarite_csm": [
    "polarite_csm_alerte",
    "polarite_csm_neutre",
    "polarite_csm_positif",
    "polarite_csm_absent"
  ],
  "niveau_anciennete": [
    "niveau_anciennete_recent",
    "niveau_anciennete_etabli",
    "niveau_anciennete_ancien"
  ]
};

// Served by the API, the page calls it on the same origin. Opened as a file,
// it targets the local service: otherwise the fetch would look for
// file:///predict and fail without a response.
const API = location.protocol.startsWith("http") ? "" : "http://127.0.0.1:8000";

// Service status, refreshed continuously: one green dot per probe that answers,
// red as soon as it fails or the service is unreachable.
const STATUS_PERIOD_MS = 5000;

const form = document.getElementById("form");
const modelSelect = document.getElementById("model");
const result = document.getElementById("result");
const error = document.getElementById("error");

for (const name of Object.keys(BOUNDS)) {
  const row = document.createElement("div");
  row.className = "field";
  const bound = BOUNDS[name];
  row.innerHTML = `<label for="${name}">${name}</label>
    <input id="${name}" name="${name}" type="number" step="${bound.integer ? 1 : 0.01}"
           placeholder="${bound.min} à ${bound.max}">`;
  form.appendChild(row);
}

const buttons = document.createElement("div");
buttons.className = "buttons";
buttons.innerHTML = `<button type="button" id="fill">Remplir</button>
  <button type="submit" class="primary">Prédire</button>`;
form.appendChild(buttons);

function drawValue(name) {
  const { min, max, integer } = BOUNDS[name];
  const value = min + Math.random() * (max - min);
  return integer ? Math.round(value) : Math.round(value * 100) / 100;
}

document.getElementById("fill").addEventListener("click", () => {
  for (const name of Object.keys(BOUNDS)) {
    document.getElementById(name).value = drawValue(name);
  }
  // One active modality per one-hot group, otherwise the row makes no sense.
  for (const modalities of Object.values(GROUPS)) {
    const active = modalities[Math.floor(Math.random() * modalities.length)];
    for (const name of modalities) {
      document.getElementById(name).value = name === active ? 1 : 0;
    }
  }
  for (const input of form.querySelectorAll("input")) input.classList.remove("missing");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.textContent = "";

  const data = {};
  const missing = [];
  for (const name of Object.keys(BOUNDS)) {
    const input = document.getElementById(name);
    input.classList.remove("missing");
    if (input.value.trim() === "") {
      input.classList.add("missing");
      missing.push(name);
    } else {
      data[name] = Number(input.value);
    }
  }

  if (missing.length) {
    error.textContent = `${missing.length} valeur(s) manquante(s) : ${missing.join(", ")}`;
    return;
  }

  try {
    const response = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: modelSelect.value, data }),
    });
    const body = await response.json();
    if (!response.ok) {
      error.textContent = `HTTP ${response.status} — ${JSON.stringify(body.detail)}`;
      return;
    }
    render(body);
  } catch (failure) {
    error.textContent = `Appel impossible : ${failure.message}\n`
      + `Cible : ${API || location.origin}/predict\n`
      + `Le service tourne-t-il ? uv run uvicorn ml_churn.api.main:app`;
  }
});

function render({ model, probability, threshold, churn }) {
  result.innerHTML = `
    <dt>model</dt><dd>${model}</dd>
    <dt>probability</dt><dd>${probability}</dd>
    <dt>threshold</dt><dd>${threshold}</dd>
    <dt>churn</dt><dd class="${churn ? "churn-true" : "churn-false"}">${churn}</dd>`;
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
}

refreshStatus();
setInterval(refreshStatus, STATUS_PERIOD_MS);
