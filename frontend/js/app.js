// app.js - Frontend interaction with FastAPI backend

const API_BASE = "/api";

// Utility to create a chat message element
function addMessage(content, role) {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", role);
  // Render bot messages as HTML to enable clickable links and line breaks
  if (role === "bot") {
    // Preserve line breaks for readability
    const html = content.replace(/\n/g, "<br>");
    msgDiv.innerHTML = html;
  } else {
    msgDiv.textContent = content;
  }
  const chatLog = document.getElementById("chat-log");
  chatLog.appendChild(msgDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
}

// Handle chat form submission
document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("user-input");
  const userMessage = input.value.trim();
  if (!userMessage) return;
  addMessage(userMessage, "user");
  input.value = "";
  try {
    const resp = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage })
    });
    const data = await resp.json();
    if (!resp.ok) {
        addMessage("Error: " + (data.error || "Unknown error"), "bot");
        return;
    }
    addMessage(data.reply, "bot");
  } catch (err) {
    console.error(err);
    addMessage("Error contacting server.", "bot");
  }
});

// File upload handling
document.getElementById("upload-btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("file-upload");
  if (!fileInput.files.length) {
    alert("Select a file first.");
    return;
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);
  try {
    const resp = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData
    });
    const data = await resp.json();
    addMessage(`File upload: ${data.reply}`, "bot");
  } catch (err) {
    console.error(err);
    addMessage("Upload failed.", "bot");
  }
});

// RAG query handling
document.getElementById("rag-btn").addEventListener("click", async () => {
  const queryInput = document.getElementById("rag-query");
  const query = queryInput.value.trim();
  if (!query) {
    alert("Enter a query.");
    return;
  }
  addMessage(query, "user");
  queryInput.value = "";
  try {
    const resp = await fetch(`${API_BASE}/rag/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: query })
    });
    const data = await resp.json();
    addMessage(data.reply, "bot");
  } catch (err) {
    console.error(err);
    addMessage("RAG query failed.", "bot");
  }
});

// Global variable to keep the latest research report
let latestResearchReport = "";

// Research handling (updated)
document.getElementById("research-btn").addEventListener("click", async () => {
  const queryInput = document.getElementById("research-query");
  const query = queryInput.value.trim();
  if (!query) { alert("Enter a research topic."); return; }
  addMessage(query, "user");
  queryInput.value = "";
  try {
    const resp = await fetch(`${API_BASE}/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const data = await resp.json();
    // Show report in chat with HTML rendering
    const reportHtml = data.report
      .replace(/\n/g, "<br>")
      .replace(/URL: (https?:\/\/[^\s]+)/g, "URL: <a href='$1' target='_blank'>$1</a>");
    addMessage(reportHtml, "bot");
    latestResearchReport = reportHtml; // store for adding to RAG
    // Show URLs if any
    if (data.urls && data.urls.length) {
      const urlsMsg = data.urls.map(u => `<a href="${u}" target="_blank">${u}</a>`).join("<br>");
      addMessage(`**Discovered URLs (${data.url_count}):**<br>${urlsMsg}`, "bot");
    }
  } catch (err) {
    console.error(err);
    addMessage("Research failed.", "bot");
  }
});

// Add Research to RAG button
document.getElementById("add-research-btn").addEventListener("click", async () => {
  if (!latestResearchReport) {
    alert("No research report available to add. Run a research query first.");
    return;
  }
  try {
    const resp = await fetch(`${API_BASE}/rag/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: latestResearchReport })
    });
    const data = await resp.json();
    addMessage(data.reply, "bot");
    // UI feedback: temporary button state
    const btn = document.getElementById("add-research-btn");
    const originalText = btn.textContent;
    btn.textContent = "Added ✔";
    btn.disabled = true;
    setTimeout(() => {
      btn.textContent = originalText;
      btn.disabled = false;
    }, 2000);
  } catch (err) {
    console.error(err);
    addMessage("Failed to add research to RAG.", "bot");
  }
});

// Mode button handling
document.getElementById("business-mode-btn").addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/mode/set`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "business" })
    });
    document.body.classList.remove("thinking-mode");
    document.body.classList.add("business-mode");
    addMessage("Business mode activated.", "bot");
  } catch (e) {
    console.error(e);
    addMessage("Failed to set business mode.", "bot");
  }
});

document.getElementById("thinking-mode-btn").addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/mode/set`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "thinking" })
    });
    document.body.classList.remove("business-mode");
    document.body.classList.add("thinking-mode");
    addMessage("Thinking mode activated.", "bot");
  } catch (e) {
    console.error(e);
    addMessage("Failed to set thinking mode.", "bot");
  }
});

// Updated New chat button to also reset mode
document.getElementById("new-chat-btn").addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/chat/reset`, { method: "POST" });
    document.getElementById("chat-log").innerHTML = "";
    // Reset mode to normal
    await fetch(`${API_BASE}/mode/set`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "normal" })
    });
    document.body.classList.remove("business-mode", "thinking-mode");
    await loadConfig();
  } catch (e) {
    console.error("Failed to reset chat", e);
  }
});

  // Load configuration and populate model selector
  async function loadConfig() {
    try {
      const cfgResp = await fetch(`${API_BASE}/config`);
      const cfg = await cfgResp.json();
      document.getElementById("model-name").textContent = cfg.model || "unknown";
      console.log('Config loaded:', cfg);
    } catch (e) {
      console.warn("Could not load config.");
    }
    // Populate model dropdown
    await loadModels();
  }

  // Load available models and set selected model
  async function loadModels() {
    try {
      const resp = await fetch(`${API_BASE}/models/installed`);
      const data = await resp.json();
      console.log('Installed models response:', data);
      const select = document.getElementById("model-select");
      // Clear existing options except placeholder
      select.innerHTML = '<option value="" disabled selected>Choose model</option>';
      data.installed_models.forEach((model) => {
        const opt = document.createElement('option');
        opt.value = model;
        opt.textContent = model;
        select.appendChild(opt);
      });
      console.log('Dropdown populated with models');
      // Set current model selection based on config
      const cfgResp = await fetch(`${API_BASE}/config`);
      const cfg = await cfgResp.json();
      if (cfg.model) {
        select.value = cfg.model;
        console.log('Selected model set to', cfg.model);
      }
    } catch (e) {
      console.warn('Failed to load models:', e);
    }
  }

  // Handle model selection change
  document.getElementById("model-select").addEventListener('change', async (e) => {
    const selectedModel = e.target.value;
    try {
      await fetch(`${API_BASE}/model/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel })
      });
      // Update displayed model name
      document.getElementById("model-name").textContent = selectedModel;
    } catch (err) {
      console.error('Failed to switch model:', err);
    }
  });

  // Initial UI setup
  loadConfig();


// New chat button clears chat log and resets session on backend
document.getElementById("new-chat-btn").addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/chat/reset`, { method: "POST" });
    document.getElementById("chat-log").innerHTML = "";
    // Optionally reset model name display
    await loadConfig();
  } catch (e) {
    console.error("Failed to reset chat", e);
  }
});

// Initial UI setup
loadConfig();
