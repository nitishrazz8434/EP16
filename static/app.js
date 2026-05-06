const sessionKey = "parkwise-session-id";
const state = {
  sessionId: localStorage.getItem(sessionKey) || crypto.randomUUID(),
  currentResultRow: null,
  refreshInFlight: false,
};

localStorage.setItem(sessionKey, state.sessionId);

const chatMessages = document.querySelector("#chatMessages");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const memoryLine = document.querySelector("#memoryLine");

appendMessage("bot", "Hlo, I am ready. Tell me where in LPU you are heading and I will find available parking.", [], [
  "LPU Main Gate",
  "Central Library",
  "Block 34",
  "Uni Mall",
  "Hostels",
  "More areas",
]);
setInterval(refreshVisibleResults, 9000);

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  messageInput.value = "";
  sendMessage(text);
});

async function sendMessage(text) {
  disableChoiceButtons();
  appendMessage("user", text);
  const waiting = appendMessage("bot", "Checking live campus availability");
  waiting.classList.add("is-typing");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: state.sessionId }),
    });
    const data = await response.json();

    state.sessionId = data.session_id;
    localStorage.setItem(sessionKey, state.sessionId);

    updateBotMessage(waiting, data.reply, data.cards || [], data.quick_replies || []);
    updateMemory(data.memory || {});
  } catch (error) {
    updateBotMessage(waiting, "The local server did not respond. Please check Flask is running.", []);
  }
}

function appendMessage(role, text, cards = [], choices = []) {
  const row = document.createElement("div");
  row.className = `message ${role}`;
  row.innerHTML = `
    <div class="avatar">${role === "user" ? "U" : "AI"}</div>
    <div class="bubble">
      <div class="message-text"></div>
      <div class="message-time">${messageTime()}</div>
    </div>
  `;
  updateBotMessage(row, text, cards, choices);
  chatMessages.appendChild(row);
  updateConversationState();
  revealMessage(row, cards.length > 0);
  return row;
}

function updateBotMessage(row, text, cards, choices = []) {
  row.classList.remove("is-typing");
  const bubble = row.querySelector(".bubble");
  const messageText = row.querySelector(".message-text");
  messageText.textContent = text;

  const oldResults = bubble.querySelector(".result-list");
  if (oldResults) oldResults.remove();
  const oldChoices = bubble.querySelector(".choice-list");
  if (oldChoices) oldChoices.remove();

  if (cards.length) {
    bubble.appendChild(createResultList(cards));
    state.currentResultRow = row;
  } else {
    state.currentResultRow = null;
  }

  if (choices.length) {
    bubble.appendChild(createChoiceList(choices));
  }
  if (row.isConnected) {
    revealMessage(row, cards.length > 0);
  }
}

function createChoiceList(choices) {
  const list = document.createElement("div");
  list.className = "choice-list";
  choices.forEach((choice) => {
    const button = document.createElement("button");
    button.className = "choice-button";
    button.type = "button";
    button.textContent = choice;
    button.addEventListener("click", () => sendMessage(choice));
    list.appendChild(button);
  });
  return list;
}

function createResultList(cards) {
  const list = document.createElement("div");
  list.className = "result-list";

  cards.forEach((lot, index) => {
    const option = document.createElement("article");
    option.className = "parking-option";
    const features = (lot.feature_labels || [])
      .slice(0, 4)
      .map((feature) => `<span>${escapeHtml(feature)}</span>`)
      .join("");
    const occupancy = Math.min(100, Math.max(0, Number(lot.occupancy_percent || 0)));
    const confidence = Math.round(Number(lot.prediction_confidence || 0) * 100);
    option.innerHTML = `
      <div class="option-top">
        <div>
          <div class="option-name">${index + 1}. ${escapeHtml(lot.name)}</div>
          <div class="option-meta">${escapeHtml(lot.area)} · ${escapeHtml(lot.reason || "balanced choice")}</div>
          <div class="option-badges">
            <span>${escapeHtml(lot.vehicle_label || "Vehicle")}</span>
            <span>${confidence}% live confidence</span>
          </div>
        </div>
        <div class="recommendation">${escapeHtml(lot.recommendation || "Option")}</div>
      </div>
      <div class="occupancy">
        <span style="width: ${occupancy}%"></span>
      </div>
      <div class="option-stats">
        <div class="stat"><span>Spaces free</span><strong>${lot.spaces_available}/${lot.capacity}</strong></div>
        <div class="stat"><span>Walk</span><strong>${lot.walking_minutes} min</strong></div>
        <div class="stat"><span>Rate</span><strong>${escapeHtml(lot.rate_label)}</strong></div>
      </div>
      <div class="features">${features}</div>
      <div class="spots">
        <span class="risk ${escapeHtml(lot.availability_risk)}">${escapeHtml(statusLabel(lot.availability_risk))}</span>
        ${lot.occupancy_percent}% occupied · updated ${escapeHtml(lot.last_updated || "recently")}
      </div>
      <button class="reserve-button" type="button">Hold this option</button>
    `;

    option.querySelector(".reserve-button").addEventListener("click", (event) => reserveLot(lot, event.currentTarget));
    list.appendChild(option);
  });

  return list;
}

async function reserveLot(lot, button) {
  button.disabled = true;
  button.textContent = "Holding...";
  const waiting = appendMessage("bot", "Holding that parking option");
  waiting.classList.add("is-typing");
  try {
    const response = await fetch("/api/reserve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        lot_id: lot.id,
        vehicle_type: lot.vehicle_type || "car",
        duration_minutes: 60,
      }),
    });
    const data = await response.json();
    updateBotMessage(waiting, data.reply || "Reservation updated.", [], data.quick_replies || []);
    button.textContent = response.ok ? "Held" : "Try again";
    if (response.ok) state.currentResultRow = null;
    if (!response.ok) button.disabled = false;
  } catch (error) {
    updateBotMessage(waiting, "I could not reserve that spot right now.", []);
    button.textContent = "Try again";
    button.disabled = false;
  }
}

async function refreshVisibleResults() {
  if (!state.currentResultRow || state.refreshInFlight) return;
  state.refreshInFlight = true;
  try {
    const response = await fetch("/api/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    const data = await response.json();
    if (data.updated && data.cards && data.cards.length && state.currentResultRow) {
      const bubble = state.currentResultRow.querySelector(".bubble");
      const oldResults = bubble.querySelector(".result-list");
      if (oldResults) oldResults.replaceWith(createResultList(data.cards));
      updateMemory(data.memory || {});
    }
  } catch (error) {
    // Keep the chat calm if a background refresh fails.
  } finally {
    state.refreshInFlight = false;
  }
}

function updateMemory(memory) {
  const bits = [];
  if (memory.vehicle_type) bits.push(label(memory.vehicle_type));
  if (memory.location) bits.push(title(memory.location));
  memoryLine.textContent = bits.length ? bits.join(" near ") : "Find available parking spots";
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function revealMessage(row, focusResults = false) {
  if (focusResults) {
    row.scrollIntoView({ block: "start", behavior: "smooth" });
    return;
  }
  scrollToBottom();
}

function disableChoiceButtons() {
  document.querySelectorAll(".choice-button").forEach((button) => {
    button.disabled = true;
  });
}

function updateConversationState() {
  chatMessages.classList.toggle("has-conversation", chatMessages.querySelectorAll(".message").length > 1);
}

function label(value) {
  const labels = {
    car: "Car",
    bike: "Bike",
    ev: "EV",
    accessible: "Accessible",
  };
  return labels[value] || title(value);
}

function title(value) {
  const normalized = String(value || "").toLowerCase();
  const replacements = {
    "lpu": "LPU",
    "lpu main gate": "LPU Main Gate",
    "lovely professional university": "Lovely Professional University",
  };
  if (replacements[normalized]) return replacements[normalized];
  return normalized
    .split(" ")
    .filter(Boolean)
    .map((part) => (part === "lpu" ? "LPU" : part.charAt(0).toUpperCase() + part.slice(1)))
    .join(" ");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function statusLabel(value) {
  const labels = {
    open: "Open",
    available: "Available",
    moderate: "Moderate",
    busy: "Busy",
    limited: "Limited",
    full: "Full",
  };
  return labels[value] || title(value);
}

function messageTime() {
  return new Intl.DateTimeFormat([], {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date());
}
