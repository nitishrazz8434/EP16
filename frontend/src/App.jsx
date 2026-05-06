import {
  BatteryCharging,
  Car,
  CheckCircle2,
  CircleParking,
  Clock3,
  IndianRupee,
  LoaderCircle,
  MapPin,
  Navigation,
  Route,
  Send,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

const SESSION_KEY = "lpu-parking-session-id";
const starterChoices = ["LPU Main Gate", "Central Library", "Block 34", "Uni Mall", "Hostels", "More areas"];

function newId() {
  return crypto.randomUUID();
}

function starterMessage() {
  return {
    id: newId(),
    role: "assistant",
    text: "Hlo, I am ready. Tell me where in LPU you are heading and I will find available parking.",
    cards: [],
    choices: starterChoices,
    pending: false,
    time: formatTime(new Date()),
  };
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => localStorage.getItem(SESSION_KEY) || newId());
  const [messages, setMessages] = useState(() => [starterMessage()]);
  const [memory, setMemory] = useState({});
  const [modelInfo, setModelInfo] = useState(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [currentResultId, setCurrentResultId] = useState(null);
  const [lastRefresh, setLastRefresh] = useState("");
  const refreshBusy = useRef(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    localStorage.setItem(SESSION_KEY, sessionId);
  }, [sessionId]);

  useEffect(() => {
    fetch("/api/model")
      .then((response) => response.json())
      .then(setModelInfo)
      .catch(() => setModelInfo(null));
  }, []);

  useEffect(() => {
    const lastWithCards = getLastMessageWithCards(messages);
    const target = lastWithCards ? document.getElementById(`message-${lastWithCards.id}`) : null;
    if (target && lastWithCards.id === currentResultId) {
      target.scrollIntoView({ block: "start", behavior: "smooth" });
      return;
    }
    chatEndRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [messages, currentResultId]);

  useEffect(() => {
    const timer = setInterval(async () => {
      if (!currentResultId || refreshBusy.current) return;
      refreshBusy.current = true;
      try {
        const response = await fetch("/api/refresh", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });
        const data = await response.json();
        if (data.updated && data.cards?.length) {
          setMessages((items) =>
            items.map((item) => (item.id === currentResultId ? { ...item, cards: data.cards } : item))
          );
          setMemory(data.memory || {});
          setLastRefresh(formatTime(new Date()));
        }
      } catch {
        // Background refresh should never interrupt the conversation.
      } finally {
        refreshBusy.current = false;
      }
    }, 9000);

    return () => clearInterval(timer);
  }, [currentResultId, sessionId]);

  const memoryLine = useMemo(() => formatMemory(memory), [memory]);
  const modelLine = modelInfo
    ? `${modelInfo.intent_model} | ${modelInfo.availability_model}`
    : "React client | Flask API | AI parking model";

  async function submitMessage(rawText) {
    const text = rawText.trim();
    if (!text || busy) return;

    setInput("");
    setBusy(true);
    const userMessage = { id: newId(), role: "user", text, cards: [], choices: [], pending: false, time: formatTime(new Date()) };
    const waitingId = newId();
    const waitingMessage = {
      id: waitingId,
      role: "assistant",
      text: "Checking live campus availability",
      cards: [],
      choices: [],
      pending: true,
      time: formatTime(new Date()),
    };

    setMessages((items) => [...items, userMessage, waitingMessage]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });
      const data = await response.json();
      const cards = data.cards || [];

      setSessionId(data.session_id || sessionId);
      setMemory(data.memory || {});
      setMessages((items) =>
        items.map((item) =>
          item.id === waitingId
            ? {
                ...item,
                text: data.reply || "I received your message.",
                cards,
                choices: data.quick_replies || [],
                pending: false,
                intent: data.intent,
                confidence: data.confidence,
              }
            : item
        )
      );
      setCurrentResultId(cards.length ? waitingId : null);
      if (cards.length) setLastRefresh(formatTime(new Date()));
    } catch {
      setMessages((items) =>
        items.map((item) =>
          item.id === waitingId
            ? {
                ...item,
                text: "The Flask API did not respond. Please check that python app.py is running.",
                pending: false,
              }
            : item
        )
      );
    } finally {
      setBusy(false);
    }
  }

  async function reserveLot(lot) {
    if (busy) return;
    setBusy(true);
    const waitingId = newId();
    setMessages((items) => [
      ...items,
      {
        id: waitingId,
        role: "assistant",
        text: `Holding ${lot.name}`,
        cards: [],
        choices: [],
        pending: true,
        time: formatTime(new Date()),
      },
    ]);

    try {
      const response = await fetch("/api/reserve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          lot_id: lot.id,
          vehicle_type: lot.vehicle_type || "car",
          duration_minutes: 60,
        }),
      });
      const data = await response.json();
      setMessages((items) =>
        items.map((item) =>
          item.id === waitingId
            ? {
                ...item,
                text: data.reply || "Reservation updated.",
                choices: data.quick_replies || [],
                pending: false,
              }
            : item
        )
      );
      if (response.ok) setCurrentResultId(null);
    } catch {
      setMessages((items) =>
        items.map((item) =>
          item.id === waitingId
            ? {
                ...item,
                text: "I could not reserve that spot right now.",
                pending: false,
              }
            : item
        )
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-frame">
      <section className="chat-shell" aria-label="LPU Parking Assistant chatbot">
        <header className="chat-header">
          <div className="brand-mark">
            <CircleParking size={24} strokeWidth={2.4} />
          </div>
          <div className="heading">
            <h1>LPU Parking Assistant</h1>
            <p>{memoryLine}</p>
            <span>{modelLine}</span>
          </div>
          <div className="model-status" title="React frontend connected to Flask API">
            <span className="pulse" />
            Ready
          </div>
        </header>

        <div className={`chat-messages ${messages.length > 1 ? "has-conversation" : ""}`} aria-live="polite">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              busy={busy}
              onChoice={submitMessage}
              onReserve={reserveLot}
            />
          ))}
          <div ref={chatEndRef} />
        </div>

        <div className="live-strip">
          <span>
            <Sparkles size={15} />
            React UI connected to Flask API
          </span>
          <span>
            <Route size={15} />
            LPU campus routing
          </span>
          <span>
            <Clock3 size={15} />
            {lastRefresh ? `Updated ${lastRefresh}` : "Live refresh ready"}
          </span>
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            submitMessage(input);
          }}
        >
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            type="text"
            autoComplete="off"
            aria-label="Message LPU Parking Assistant"
            placeholder="Ask parking or AI: car near library, cheapest near Uni Mall, what is ensemble learning"
          />
          <button type="submit" disabled={busy || !input.trim()} aria-label="Send message">
            {busy ? <LoaderCircle className="spin" size={20} /> : <Send size={19} />}
            <span>Send</span>
          </button>
        </form>
      </section>
    </main>
  );
}

function ChatMessage({ message, busy, onChoice, onReserve }) {
  const isUser = message.role === "user";
  return (
    <article id={`message-${message.id}`} className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="avatar">{isUser ? "U" : "AI"}</div>
      <div className="bubble">
        <p className={message.pending ? "message-text pending" : "message-text"}>{message.text}</p>
        {message.intent && (
          <div className="intent-chip">
            <Sparkles size={13} />
            {intentLabel(message.intent)} {message.confidence ? `${Math.round(message.confidence * 100)}%` : ""}
          </div>
        )}
        {message.cards?.length > 0 && (
          <div className="result-list">
            {message.cards.map((lot, index) => (
              <ParkingCard key={`${lot.id}-${lot.vehicle_type}`} lot={lot} index={index} onReserve={onReserve} busy={busy} />
            ))}
          </div>
        )}
        {message.choices?.length > 0 && (
          <div className="choice-list">
            {message.choices.map((choice) => (
              <button key={choice} className="choice-button" type="button" onClick={() => onChoice(choice)} disabled={busy}>
                {choice}
              </button>
            ))}
          </div>
        )}
        <time>{message.time}</time>
      </div>
    </article>
  );
}

function ParkingCard({ lot, index, onReserve, busy }) {
  const occupancy = Math.min(100, Math.max(0, Number(lot.occupancy_percent || 0)));
  const confidence = Math.round(Number(lot.prediction_confidence || 0) * 100);

  return (
    <article className="parking-card">
      <div className="card-top">
        <div>
          <p className="option-number">Option {index + 1}</p>
          <h2>{lot.name}</h2>
          <p className="card-meta">
            <MapPin size={14} />
            {lot.area} | {lot.reason || "balanced option"}
          </p>
        </div>
        <span className="recommendation">{lot.recommendation || "Recommended"}</span>
      </div>

      <div className="meter" aria-label={`${lot.occupancy_percent}% occupied`}>
        <span style={{ width: `${occupancy}%` }} />
      </div>

      <div className="stats">
        <Stat icon={<CircleParking size={16} />} label="Spaces free" value={`${lot.spaces_available}/${lot.capacity}`} />
        <Stat icon={<Navigation size={16} />} label="Walk" value={`${lot.walking_minutes} min`} />
        <Stat icon={<IndianRupee size={16} />} label="Rate" value={lot.rate_label} />
      </div>

      <div className="card-badges">
        <span>
          <Car size={14} />
          {lot.vehicle_label || "Vehicle"}
        </span>
        <span>
          <ShieldCheck size={14} />
          {confidence}% confidence
        </span>
        <span className={`risk ${lot.availability_risk}`}>{statusLabel(lot.availability_risk)}</span>
        {lot.features?.includes("ev_charging") && (
          <span>
            <BatteryCharging size={14} />
            EV charging
          </span>
        )}
      </div>

      <div className="features">
        {(lot.feature_labels || []).slice(0, 5).map((feature) => (
          <span key={feature}>{feature}</span>
        ))}
      </div>

      <div className="card-footer">
        <span>
          <Clock3 size={14} />
          {lot.occupancy_percent}% occupied | updated {lot.last_updated || "recently"}
        </span>
        <button type="button" onClick={() => onReserve(lot)} disabled={busy || lot.availability_risk === "full"}>
          <CheckCircle2 size={17} />
          Hold this option
        </button>
      </div>
    </article>
  );
}

function Stat({ icon, label, value }) {
  return (
    <div className="stat">
      <span>
        {icon}
        {label}
      </span>
      <strong>{value}</strong>
    </div>
  );
}

function getLastMessageWithCards(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].cards?.length) return messages[index];
  }
  return null;
}

function formatMemory(memory) {
  const vehicle = memory.vehicle_type ? titleCase(memory.vehicle_type) : "";
  const location = memory.location ? titleCase(memory.location) : "";
  if (vehicle && location) return `${vehicle} near ${location}`;
  if (location) return `Checking ${location}`;
  return "Find available parking spots at Lovely Professional University";
}

function titleCase(value) {
  const normalized = String(value || "").toLowerCase();
  const replacements = {
    lpu: "LPU",
    ev: "EV",
    "lpu main gate": "LPU Main Gate",
    "lovely professional university": "Lovely Professional University",
    "academic block 34": "Block 34",
  };
  if (replacements[normalized]) return replacements[normalized];
  return normalized
    .split(" ")
    .filter(Boolean)
    .map((part) => (part === "lpu" ? "LPU" : part.charAt(0).toUpperCase() + part.slice(1)))
    .join(" ");
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
  return labels[value] || titleCase(value);
}

function intentLabel(value) {
  return String(value || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatTime(date) {
  return new Intl.DateTimeFormat([], {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}
