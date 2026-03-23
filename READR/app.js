const API_BASE = "http://localhost:8000";
const PROCESSING_DELAY_MS = 800;

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const processBtn = document.getElementById("processBtn");
const processingSteps = document.getElementById("processingSteps");
const errorBox = document.getElementById("errorBox");
const uploadScreen = document.getElementById("uploadScreen");
const resultsScreen = document.getElementById("resultsScreen");
const newDocumentBtn = document.getElementById("newDocumentBtn");
const openChatBtn = document.getElementById("openChatBtn");
const summaryCard = document.getElementById("summaryCard");
const insightsList = document.getElementById("insightsList");
const namesTags = document.getElementById("namesTags");
const orgsTags = document.getElementById("orgsTags");
const datesTags = document.getElementById("datesTags");
const valuesTags = document.getElementById("valuesTags");
const chatOverlay = document.getElementById("chatOverlay");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatFilename = document.getElementById("chatFilename");
const closeChatBtn = document.getElementById("closeChatBtn");

let selectedFile = null;
let currentDocId = "";
let currentFilename = "";
let typingIndicator = null;

function showError(message) {
  errorBox.textContent = `ERROR: ${message}`;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function setSelectedFile(file) {
  selectedFile = file;
  currentFilename = file ? file.name : "";
  fileName.textContent = file ? file.name : "NO FILE SELECTED";
  dropZone.classList.toggle("file-selected", Boolean(file));
}

function resetSteps() {
  Array.from(processingSteps.querySelectorAll("p")).forEach((step) => {
    step.classList.remove("active", "done");
  });
}

async function animateSteps() {
  processingSteps.classList.remove("hidden");
  const steps = Array.from(processingSteps.querySelectorAll("p"));
  for (const [index, step] of steps.entries()) {
    step.classList.add("active");
    await new Promise((resolve) => window.setTimeout(resolve, PROCESSING_DELAY_MS));
    if (index < steps.length - 1) {
      step.classList.remove("active");
      step.classList.add("done");
    }
  }
}

function finishSteps() {
  const steps = Array.from(processingSteps.querySelectorAll("p"));
  steps.forEach((step) => {
    step.classList.remove("active");
    step.classList.add("done");
  });
}

function renderTags(container, items) {
  container.innerHTML = "";
  const values = Array.isArray(items) && items.length ? items : ["NONE"];
  values.forEach((item, index) => {
    const tag = document.createElement("span");
    tag.className = `entity-tag${index % 2 === 1 ? " alt" : ""}`;
    tag.textContent = item;
    container.appendChild(tag);
  });
}

function renderInsights(items) {
  insightsList.innerHTML = "";
  const values = Array.isArray(items) && items.length ? items : ["No insights returned."];
  values.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    insightsList.appendChild(li);
  });
}

function renderResults(data) {
  summaryCard.textContent = data.summary || "No summary returned.";
  renderTags(namesTags, data.entities?.names);
  renderTags(orgsTags, data.entities?.organisations);
  renderTags(datesTags, data.entities?.dates);
  renderTags(valuesTags, data.entities?.values);
  renderInsights(data.insights);
}

function showResults() {
  uploadScreen.classList.add("hidden");
  resultsScreen.classList.remove("hidden");
  newDocumentBtn.classList.remove("hidden");
}

function resetAppState() {
  currentDocId = "";
  currentFilename = "";
  setSelectedFile(null);
  clearError();
  processBtn.disabled = false;
  processBtn.textContent = "PROCESS DOCUMENT";
  processingSteps.classList.add("hidden");
  resetSteps();
  uploadScreen.classList.remove("hidden");
  resultsScreen.classList.add("hidden");
  newDocumentBtn.classList.add("hidden");
  closeChatOverlay();
  clearChatMessages();
}

async function readResponse(response) {
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    return { detail: text || `Request failed with status ${response.status}` };
  }
}

async function processDocument() {
  clearError();
  if (!selectedFile) {
    showError("Please choose a PDF or TXT file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  processBtn.disabled = true;
  processBtn.textContent = "PROCESSING...";
  resetSteps();

  try {
    const animationPromise = animateSteps();
    const response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await readResponse(response);
    await animationPromise;

    if (!response.ok) {
      throw new Error(data.detail || "Unable to process document.");
    }

    finishSteps();
    currentDocId = data.doc_id || "";
    renderResults(data);
    showResults();
  } catch (error) {
    processingSteps.classList.add("hidden");
    showError(error.message || "Unable to process document.");
  } finally {
    processBtn.disabled = false;
    processBtn.textContent = "PROCESS DOCUMENT";
  }
}

function clearChatMessages() {
  chatMessages.innerHTML = "";
}

function scrollChatToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendMessage(role, text, isTyping = false) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (role === "assistant" && !isTyping) {
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = "READR";
    bubble.appendChild(label);
  }

  if (role === "assistant" && isTyping) {
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = "READR";
    bubble.appendChild(label);
  }

  const content = document.createElement("div");
  content.className = "message-content";
  if (isTyping) {
    const indicator = document.createElement("div");
    indicator.className = "typing-indicator";
    indicator.innerHTML = "<span></span><span></span><span></span>";
    content.appendChild(indicator);
  } else {
    content.textContent = text;
  }

  bubble.appendChild(content);
  row.appendChild(bubble);
  chatMessages.appendChild(row);
  scrollChatToBottom();
  return row;
}

function showTyping() {
  typingIndicator = appendMessage("assistant", "", true);
}

function hideTyping() {
  if (typingIndicator) {
    typingIndicator.remove();
    typingIndicator = null;
  }
}

async function clearChatSession() {
  try {
    await fetch(`${API_BASE}/new-chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ session_id: "default" }),
    });
  } catch {
    // Keep UI responsive even if clearing server-side history fails.
  }
}

async function openChatOverlay() {
  clearError();
  await clearChatSession();
  clearChatMessages();
  chatFilename.textContent = currentFilename || "DOCUMENT";
  chatOverlay.classList.remove("hidden");
  chatOverlay.setAttribute("aria-hidden", "false");
  appendMessage("assistant", "READR here. I can summarize, extract details, and surface insights from this document. Ask me anything.");
  chatInput.focus();
}

function closeChatOverlay() {
  chatOverlay.classList.add("hidden");
  chatOverlay.setAttribute("aria-hidden", "true");
}

async function submitQuestion(event) {
  event.preventDefault();
  clearError();

  const question = chatInput.value.trim();
  if (!question || !currentDocId) {
    return;
  }

  appendMessage("user", question);
  chatInput.value = "";
  showTyping();

  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        doc_id: currentDocId,
        question,
      }),
    });
    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Unable to answer question.");
    }

    hideTyping();
    appendMessage("assistant", data.answer || "Not found in document.");
  } catch (error) {
    hideTyping();
    appendMessage("assistant", error.message || "Unable to answer question.");
    showError(error.message || "Unable to answer question.");
  }
}

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  setSelectedFile(file || null);
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
  const [file] = event.dataTransfer.files;
  setSelectedFile(file || null);
});

processBtn.addEventListener("click", processDocument);
newDocumentBtn.addEventListener("click", resetAppState);
openChatBtn.addEventListener("click", openChatOverlay);
closeChatBtn.addEventListener("click", closeChatOverlay);
chatForm.addEventListener("submit", submitQuestion);

resetAppState();
