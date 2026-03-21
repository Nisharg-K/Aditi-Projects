const fileInput = document.getElementById("fileInput");
const pickFileBtn = document.getElementById("pickFileBtn");
const uploadBtn = document.getElementById("uploadBtn");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const insightsList = document.getElementById("insightsList");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");
const llamaModelBtn = document.getElementById("llamaModelBtn");
const qwenModelBtn = document.getElementById("qwenModelBtn");
const modelBadge = document.getElementById("modelBadge");
const modelStatus = document.getElementById("modelStatus");
const ragModeBtn = document.getElementById("ragModeBtn");
const directModeBtn = document.getElementById("directModeBtn");
const modeBadge = document.getElementById("modeBadge");
const modeStatus = document.getElementById("modeStatus");
const modeWarning = document.getElementById("modeWarning");
const newChatBtn = document.getElementById("newChatBtn");
const uploadProgress = document.getElementById("uploadProgress");
const uploadModeNote = document.getElementById("uploadModeNote");
const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");
const contextModal = document.getElementById("contextModal");
const contextContent = document.getElementById("contextContent");
const closeContextBtn = document.getElementById("closeContextBtn");
const promptDebugList = document.getElementById("promptDebugList");

const entityTargets = {
  names: document.getElementById("namesList"),
  orgs: document.getElementById("orgsList"),
  dates: document.getElementById("datesList"),
  values: document.getElementById("valuesList"),
};

const LLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M";
const QWEN_MODEL = "qwen3-vl:8b";
const RAG_MODE = "rag";
const DIRECT_MODE = "direct";
const sessionId = `session-${Date.now()}`;

let selectedFile = null;
let currentContext = "";
let currentModel = LLAMA_MODEL;
let currentMode = RAG_MODE;
let currentWordCount = 0;
let hasProcessedDocument = false;
let typingIndicator = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#8f1f1f" : "";
}

async function readResponse(response) {
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    return { detail: text || `Request failed with status ${response.status}` };
  }
}

function scrollChatToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatTime(date = new Date()) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDebugExtra(extra = {}) {
  const entries = Object.entries(extra).filter(([, value]) => value !== undefined && value !== null && value !== "");
  if (entries.length === 0) {
    return "";
  }
  return entries.map(([key, value]) => `${key}: ${value}`).join(" | ");
}

function renderPromptLogs(logs) {
  promptDebugList.innerHTML = "";
  if (!logs || logs.length === 0) {
    promptDebugList.textContent = "Upload a document or ask a question to see the prompts sent to Ollama.";
    return;
  }

  const orderedLogs = [...logs].reverse();
  orderedLogs.forEach((log) => {
    const entry = document.createElement("details");
    entry.className = "prompt-entry";

    const summary = document.createElement("summary");
    summary.className = "prompt-entry-summary";
    summary.textContent = log.title || "Prompt";

    const meta = formatDebugExtra(log.extra);
    if (meta) {
      const metaSpan = document.createElement("span");
      metaSpan.className = "prompt-entry-meta";
      metaSpan.textContent = meta;
      summary.appendChild(metaSpan);
    }

    const body = document.createElement("pre");
    body.className = "prompt-entry-body";
    body.textContent = log.content || "";

    entry.appendChild(summary);
    entry.appendChild(body);
    promptDebugList.appendChild(entry);
  });
}

async function loadPromptLogs() {
  try {
    const response = await fetch("http://localhost:8000/debug/prompts");
    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Unable to load prompt logs.");
    }

    renderPromptLogs(data.logs || []);
  } catch (error) {
    promptDebugList.textContent = error.message || "Unable to load prompt logs.";
  }
}

function setModelState(model, message = "") {
  currentModel = model;
  const isQwen = model === QWEN_MODEL;

  llamaModelBtn.classList.toggle("active", model === LLAMA_MODEL);
  qwenModelBtn.classList.toggle("active", isQwen);
  modelBadge.textContent = isQwen ? "OCR mode - works on scanned PDFs" : "Text mode - digital PDFs";
  modelStatus.textContent = message;
}

function updateModeWarning() {
  const showWarning = hasProcessedDocument && currentMode === DIRECT_MODE && currentWordCount > 2000;
  modeWarning.classList.toggle("hidden", !showWarning);
}

function setModeState(mode, message = "") {
  currentMode = mode;
  const isDirect = mode === DIRECT_MODE;

  ragModeBtn.classList.toggle("active", mode === RAG_MODE);
  directModeBtn.classList.toggle("active", isDirect);
  modeBadge.textContent = isDirect ? "Full document in context" : "Vector search + reranking";
  modeStatus.textContent = message;
  updateModeWarning();
}

function renderList(target, items, fallback) {
  target.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = fallback;
    target.appendChild(li);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  });
}

function openContextModal(context) {
  contextContent.textContent = context || "No context was returned for this answer.";
  contextModal.classList.remove("hidden");
}

function closeContextModal() {
  contextModal.classList.add("hidden");
}

function appendMessage(role, text, timestamp = formatTime(), isTyping = false, context = "") {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;

  if (role === "assistant") {
    const icon = document.createElement("span");
    icon.className = "bot-icon";
    icon.textContent = "AI";
    bubble.appendChild(icon);
  }

  const body = document.createElement("div");
  body.className = "message-body";

  if (isTyping) {
    body.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  } else {
    body.textContent = text;
  }

  const meta = document.createElement("div");
  meta.className = "message-time";
  meta.textContent = timestamp;

  if (role === "assistant" && !isTyping && context) {
    const infoBtn = document.createElement("button");
    infoBtn.className = "context-info-btn";
    infoBtn.type = "button";
    infoBtn.textContent = "i";
    infoBtn.title = "Show context sent to the LLM";
    infoBtn.addEventListener("click", () => openContextModal(context));
    bubble.appendChild(infoBtn);
  }

  bubble.appendChild(body);
  bubble.appendChild(meta);
  row.appendChild(bubble);
  chatMessages.appendChild(row);
  scrollChatToBottom();
  return row;
}

function showTypingIndicator() {
  typingIndicator = appendMessage("assistant", "", formatTime(), true);
}

function removeTypingIndicator() {
  if (typingIndicator) {
    typingIndicator.remove();
    typingIndicator = null;
  }
}

function selectFile(file) {
  selectedFile = file;
  fileName.textContent = file ? file.name : "No file selected";
}

function resetProgress() {
  uploadProgress.classList.add("hidden");
  uploadModeNote.textContent = "";
  step2.textContent =
    currentMode === DIRECT_MODE ? "Step 2: Skipping vector indexing in Direct mode..." : "Step 2: Chunking and indexing...";
  [step1, step2, step3].forEach((step) => {
    step.classList.remove("active", "done");
  });
}

async function runUploadProgress() {
  uploadProgress.classList.remove("hidden");
  uploadModeNote.textContent =
    currentModel === QWEN_MODEL && selectedFile?.name.toLowerCase().endsWith(".pdf")
      ? "Scanning document with OCR... this may take a moment"
      : "";
  step2.textContent =
    currentMode === DIRECT_MODE ? "Step 2: Skipping vector indexing in Direct mode..." : "Step 2: Chunking and indexing...";

  step1.classList.add("active");
  await new Promise((resolve) => setTimeout(resolve, 350));
  step1.classList.add("done");
  step1.classList.remove("active");
  step2.classList.add("active");

  await new Promise((resolve) => setTimeout(resolve, 350));
  step2.classList.add("done");
  step2.classList.remove("active");
  step3.classList.add("active");
}

async function uploadFile() {
  if (!selectedFile) {
    setStatus("Choose a PDF or TXT file first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  uploadBtn.disabled = true;
  setStatus("Processing document with the local model...");
  resetProgress();

  try {
    const progressPromise = runUploadProgress();
    const response = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData,
    });

    const data = await readResponse(response);
    await progressPromise;
    if (!response.ok) {
      throw new Error(data.detail || "Upload failed.");
    }

    currentContext = data.context || "";
    currentWordCount = data.word_count || 0;
    hasProcessedDocument = true;
    summaryEl.textContent = data.summary || "No summary returned.";
    renderList(entityTargets.names, data.entities?.names, "No names found.");
    renderList(entityTargets.orgs, data.entities?.orgs, "No organizations found.");
    renderList(entityTargets.dates, data.entities?.dates, "No dates found.");
    renderList(entityTargets.values, data.entities?.values, "No values found.");
    renderList(insightsList, data.insights, "No insights returned.");
    step3.classList.add("done");
    step3.classList.remove("active");
    updateModeWarning();
    await loadPromptLogs();
    setStatus("Document processed successfully.");
  } catch (error) {
    setStatus(error.message || "Something went wrong during upload.", true);
    resetProgress();
  } finally {
    uploadBtn.disabled = false;
  }
}

async function switchModel(model) {
  try {
    const response = await fetch("http://localhost:8000/set-model", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model }),
    });

    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Unable to switch model.");
    }

    const label = model === QWEN_MODEL ? "Qwen3 VL 8B" : "Llama 3.1 8B";
    hasProcessedDocument = false;
    currentContext = "";
    resetProgress();
    setModelState(data.model, `Switched to ${label}`);
    await loadPromptLogs();
  } catch (error) {
    setStatus(error.message || "Unable to switch model.", true);
  }
}

async function switchMode(mode) {
  try {
    const response = await fetch("http://localhost:8000/set-mode", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ mode }),
    });

    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Unable to switch mode.");
    }

    const label = data.mode === DIRECT_MODE ? "Direct Mode" : "RAG Mode";
    resetProgress();
    setModeState(data.mode, `Switched to ${label}`);
    setStatus(`Switched to ${label}`);
    await loadPromptLogs();
  } catch (error) {
    const message = error.message || "Unable to switch mode.";
    const hint =
      message.includes("404") || message.includes("Not Found")
        ? "Mode switch is unavailable on the backend. Restart the server and refresh the page."
        : message;
    modeStatus.textContent = "Backend mode route unavailable";
    setStatus(hint, true);
  }
}

async function loadCurrentModel() {
  try {
    const response = await fetch("http://localhost:8000/current-model");
    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Unable to load active model.");
    }

    setModelState(data.model);
  } catch (error) {
    setStatus(error.message || "Unable to load active model.", true);
  }
}

async function loadCurrentMode() {
  try {
    const response = await fetch("http://localhost:8000/current-mode");
    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Unable to load active mode.");
    }

    setModeState(data.mode);
  } catch (error) {
    const message = error.message || "Unable to load active mode.";
    const hint =
      message.includes("404") || message.includes("Not Found")
        ? "Mode controls need the updated backend. Restart the server and refresh the page."
        : message;
    modeStatus.textContent = "Backend mode route unavailable";
    setStatus(hint, true);
  }
}

async function clearChat() {
  try {
    const response = await fetch("http://localhost:8000/new-chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ session_id: sessionId }),
    });

    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Unable to start a new chat.");
    }

    chatMessages.innerHTML = "";
    appendMessage("assistant", "Ask a question after uploading a document and I'll answer from that context.");
    setStatus("Started a new chat.");
  } catch (error) {
    setStatus(error.message || "Unable to start a new chat.", true);
  }
}

pickFileBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => selectFile(event.target.files[0]));
uploadBtn.addEventListener("click", uploadFile);
llamaModelBtn.addEventListener("click", () => switchModel(LLAMA_MODEL));
qwenModelBtn.addEventListener("click", () => switchModel(QWEN_MODEL));
ragModeBtn.addEventListener("click", () => switchMode(RAG_MODE));
directModeBtn.addEventListener("click", () => switchMode(DIRECT_MODE));
newChatBtn.addEventListener("click", clearChat);
closeContextBtn.addEventListener("click", closeContextModal);
contextModal.addEventListener("click", (event) => {
  if (event.target === contextModal) {
    closeContextModal();
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    selectFile(file);
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = chatInput.value.trim();

  if (!query) {
    return;
  }

  if (!hasProcessedDocument) {
    setStatus("Upload and process a document before asking questions.", true);
    return;
  }

  appendMessage("user", query);
  chatInput.value = "";
  showTypingIndicator();
  setStatus("Generating grounded answer...");

  try {
    const response = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, context: currentContext, session_id: sessionId }),
    });

    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Query failed.");
    }

    removeTypingIndicator();
    appendMessage("assistant", data.answer || "No answer returned.", formatTime(), false, data.retrieved_context || "");
    await loadPromptLogs();
    setStatus("Answer ready.");
  } catch (error) {
    removeTypingIndicator();
    appendMessage("assistant", error.message || "Unable to answer the question.");
    setStatus(error.message || "Unable to answer the question.", true);
  }
});

appendMessage("assistant", "Ask a question after uploading a document and I'll answer from that context.");
loadCurrentModel();
loadCurrentMode();
loadPromptLogs();
