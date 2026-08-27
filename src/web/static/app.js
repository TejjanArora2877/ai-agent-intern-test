/**
 * Aster & Row Support Agent — Frontend JavaScript Controller
 * Clean, lightweight, modular client interacting with the FastAPI backend.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const typingIndicator = document.getElementById("typing-indicator");
  const modeSelect = document.getElementById("mode-select");
  const liveBadge = document.getElementById("live-badge");
  const sessionIdText = document.getElementById("session-id-text");
  const sessionBadgeBtn = document.getElementById("session-badge-btn");
  const newChatBtn = document.getElementById("new-chat-btn");
  const toast = document.getElementById("toast");
  const toastMessage = document.getElementById("toast-message");
  const toastClose = document.getElementById("toast-close");

  // State
  let sessionId = sessionStorage.getItem("aster_row_session_id") || "";
  let isLoading = false;
  let liveLLMConfigured = false;

  // Initialize
  initApp();

  async function initApp() {
    await fetchHealth();
    if (!sessionId) {
      await startNewSession(false);
    } else {
      updateSessionDisplay();
    }

    setupEventListeners();
  }

  function setupEventListeners() {
    // Form submission
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleSendMessage();
    });

    // Enter key handling (Enter to send, Shift+Enter for newline)
    userInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    // Auto-resize textarea
    userInput.addEventListener("input", () => {
      userInput.style.height = "auto";
      userInput.style.height = Math.min(userInput.scrollHeight, 140) + "px";
    });

    // Mode change
    modeSelect.addEventListener("change", () => {
      const mode = modeSelect.value;
      if (mode === "live") {
        liveBadge.textContent = liveLLMConfigured ? "Live Gemini" : "Live (No Key)";
        liveBadge.className = liveLLMConfigured ? "status-badge badge-live" : "status-badge badge-offline";
        if (!liveLLMConfigured) {
          showToast("Warning: GEMINI_API_KEY is not configured on the server. Live mode may fall back safely.", 5000);
        }
      } else {
        liveBadge.textContent = "Offline Mode";
        liveBadge.className = "status-badge badge-offline";
      }
    });

    // New Chat Button
    newChatBtn.addEventListener("click", () => {
      startNewSession(true);
    });

    // Copy Session ID
    sessionBadgeBtn.addEventListener("click", () => {
      if (sessionId) {
        navigator.clipboard.writeText(sessionId).then(() => {
          showToast(`Copied Session ID: ${sessionId}`, 2500);
        }).catch(() => {
          showToast(`Session ID: ${sessionId}`, 3000);
        });
      }
    });

    // Toast Close
    toastClose.addEventListener("click", () => {
      toast.classList.add("hidden");
    });

    // Sample Query Chips
    document.querySelectorAll(".sample-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const query = chip.getAttribute("data-query");
        if (query && !isLoading) {
          userInput.value = query;
          handleSendMessage();
        }
      });
    });
  }

  async function fetchHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        liveLLMConfigured = Boolean(data.live_llm_configured);
      }
    } catch (err) {
      console.warn("Could not connect to health endpoint:", err);
    }
  }

  async function startNewSession(clearUI = true) {
    try {
      const res = await fetch("/api/session/new", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        sessionId = data.session_id;
        sessionStorage.setItem("aster_row_session_id", sessionId);
        updateSessionDisplay();

        if (clearUI) {
          resetChatUI();
          showToast("Started new chat session", 2500);
        }
      }
    } catch (err) {
      sessionId = `web_${Math.random().toString(36).substring(2, 12)}`;
      sessionStorage.setItem("aster_row_session_id", sessionId);
      updateSessionDisplay();
    }
  }

  function updateSessionDisplay() {
    sessionIdText.textContent = sessionId;
  }

  function resetChatUI() {
    chatMessages.innerHTML = `
      <div class="welcome-card">
        <div class="welcome-icon">👋</div>
        <div class="welcome-content">
          <h3>Welcome to Aster &amp; Row Support</h3>
          <p>I can help you check return policies, domestic &amp; international shipping timelines, product warranty coverage, product care guidelines, or track your orders.</p>
          <div class="sample-queries">
            <span class="sample-label">Try asking:</span>
            <button class="sample-chip" data-query="What is your return window for standard orders?">Return Policy</button>
            <button class="sample-chip" data-query="Where is ORD-1007 and when will it arrive?">Track ORD-1007</button>
            <button class="sample-chip" data-query="Do you ship internationally to Canada?">Canada Shipping</button>
            <button class="sample-chip" data-query="Can I put the Breeze Tumbler in the dishwasher?">Tumbler Care</button>
          </div>
        </div>
      </div>
    `;

    // Rebind sample chips in new welcome card
    chatMessages.querySelectorAll(".sample-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const query = chip.getAttribute("data-query");
        if (query && !isLoading) {
          userInput.value = query;
          handleSendMessage();
        }
      });
    });
  }

  async function handleSendMessage() {
    const message = userInput.value.trim();
    if (!message || isLoading) return;

    // Append User Message to UI
    appendUserMessage(message);
    userInput.value = "";
    userInput.style.height = "auto";

    // Set Loading State
    setLoadingState(true);

    const mode = modeSelect.value || "offline";

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: message,
          session_id: sessionId,
          mode: mode,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      if (data.session_id) {
        sessionId = data.session_id;
        sessionStorage.setItem("aster_row_session_id", sessionId);
        updateSessionDisplay();
      }

      appendAgentMessage(data);
    } catch (err) {
      appendErrorMessage(err.message || "Failed to communicate with support agent.");
      showToast(`Error: ${err.message}`, 4000);
    } finally {
      setLoadingState(false);
    }
  }

  function setLoadingState(loading) {
    isLoading = loading;
    sendBtn.disabled = loading;
    userInput.disabled = loading;

    if (loading) {
      typingIndicator.classList.remove("hidden");
      scrollToBottom();
    } else {
      typingIndicator.classList.add("hidden");
      userInput.focus();
    }
  }

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user-row";

    const bubbleWrapper = document.createElement("div");
    bubbleWrapper.className = "message-bubble-wrapper";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble user-bubble";
    bubble.textContent = text;

    bubbleWrapper.appendChild(bubble);

    const avatar = document.createElement("div");
    avatar.className = "avatar user-avatar";
    avatar.textContent = "You";

    row.appendChild(bubbleWrapper);
    row.appendChild(avatar);

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function appendAgentMessage(data) {
    const row = document.createElement("div");
    row.className = "message-row agent-row";

    const avatar = document.createElement("div");
    avatar.className = "avatar agent-avatar";
    avatar.textContent = "A&R";

    const bubbleWrapper = document.createElement("div");
    bubbleWrapper.className = "message-bubble-wrapper";

    // 1. Main Answer Bubble
    const bubble = document.createElement("div");
    bubble.className = "message-bubble agent-bubble";
    bubble.innerHTML = formatMarkdownText(data.answer || "");

    bubbleWrapper.appendChild(bubble);

    // 2. Human Specialist Handoff Banner (if applicable)
    if (data.handoff) {
      const handoffBanner = document.createElement("div");
      handoffBanner.className = "handoff-banner";
      handoffBanner.innerHTML = `
        <svg class="handoff-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
          <line x1="12" y1="9" x2="12" y2="13"></line>
          <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
        <span>Human Specialist Review Recommended</span>
      `;
      bubbleWrapper.appendChild(handoffBanner);
    }

    // 3. Source Citations (if present)
    if (data.sources && data.sources.length > 0) {
      const citationsContainer = document.createElement("div");
      citationsContainer.className = "citations-container";

      const label = document.createElement("span");
      label.className = "citations-label";
      label.textContent = "Sources:";
      citationsContainer.appendChild(label);

      data.sources.forEach((src) => {
        const pill = document.createElement("span");
        pill.className = "source-pill";
        pill.textContent = `${src.file} > ${src.heading}`;
        citationsContainer.appendChild(pill);
      });

      bubbleWrapper.appendChild(citationsContainer);
    }

    // 4. Expandable Debug / Trace Accordion
    if (data.debug_trace) {
      const trace = data.debug_trace;
      const debugDetails = document.createElement("details");
      debugDetails.className = "debug-details";

      const latency = trace.latency_ms ? `${trace.latency_ms.toFixed(1)}ms` : "N/A";
      const modelMode = trace.model_mode || "mock";

      debugDetails.innerHTML = `
        <summary class="debug-summary">
          <span>🔍 Debug Observability Trace</span>
          <div class="debug-meta-pills">
            <span class="debug-pill">${modelMode}</span>
            <span class="debug-pill">${latency}</span>
          </div>
        </summary>
        <div class="debug-body">
          <div class="debug-row">
            <span class="debug-label">Order Query:</span>
            <span class="debug-val">${trace.order_query_detected ? `Yes (ID: ${trace.order_id_extracted || 'None'})` : 'No'}</span>
          </div>
          ${trace.order_tool_result ? `
            <div class="debug-row">
              <span class="debug-label">Sanitized Order:</span>
              <span class="debug-val">Status: <strong>${trace.order_tool_result.status || 'N/A'}</strong> ${trace.order_tool_result.carrier ? `| Carrier: ${trace.order_tool_result.carrier}` : ''}</span>
            </div>
          ` : ''}
          ${data.tool_calls && data.tool_calls.length > 0 ? `
            <div class="debug-row">
              <span class="debug-label">Tool Execution:</span>
              <span class="debug-val">${data.tool_calls.map(tc => `${tc.tool_name}(${JSON.stringify(tc.arguments)})`).join(', ')}</span>
            </div>
          ` : ''}
          <div class="debug-row">
            <span class="debug-label">Conflict Detected:</span>
            <span class="debug-val">${trace.conflict_detected ? '⚠️ Yes' : 'No'}</span>
          </div>
          ${trace.retrieved_chunks && trace.retrieved_chunks.length > 0 ? `
            <div>
              <span class="debug-label">Retrieved Chunks (${trace.retrieved_chunks.length}):</span>
              <div class="debug-chunks-list">
                ${trace.retrieved_chunks.map((c, i) => `
                  <div class="debug-chunk-item">
                    <span>[${i+1}] ${escapeHtml(c.file_name || '')} &gt; <em>${escapeHtml(c.heading || '')}</em></span>
                    <span class="chunk-score">${c.score ? c.score.toFixed(2) : '0.00'}</span>
                  </div>
                `).join('')}
              </div>
            </div>
          ` : '<div class="debug-row"><span class="debug-label">Retrieved Chunks:</span><span class="debug-val">0 (None needed)</span></div>'}
        </div>
      `;

      bubbleWrapper.appendChild(debugDetails);
    }

    row.appendChild(avatar);
    row.appendChild(bubbleWrapper);

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function appendErrorMessage(errorText) {
    const row = document.createElement("div");
    row.className = "message-row agent-row";

    const avatar = document.createElement("div");
    avatar.className = "avatar agent-avatar";
    avatar.style.background = "#ef4444";
    avatar.textContent = "!";

    const bubbleWrapper = document.createElement("div");
    bubbleWrapper.className = "message-bubble-wrapper";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble agent-bubble";
    bubble.style.borderColor = "#ef4444";
    bubble.innerHTML = `<strong style="color:#f87171;">System Error:</strong> ${escapeHtml(errorText)}`;

    bubbleWrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(bubbleWrapper);

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function scrollToBottom() {
    const workspace = document.querySelector(".chat-workspace");
    if (workspace) {
      workspace.scrollTop = workspace.scrollHeight;
    }
  }

  function formatMarkdownText(text) {
    if (!text) return "";
    let safe = escapeHtml(text);
    // Bold formatting: **text** -> <strong>text</strong>
    safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Italic formatting: *text* -> <em>$1</em>
    safe = safe.replace(/\*(.*?)\*/g, "<em>$1</em>");
    // Bullet points: lines starting with "- " or "* "
    safe = safe.replace(/^(?:-|\*)\s+(.+)$/gm, "• $1");
    // Line breaks
    safe = safe.replace(/\n/g, "<br>");
    return safe;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function showToast(message, duration = 3000) {
    toastMessage.textContent = message;
    toast.classList.remove("hidden");
    setTimeout(() => {
      toast.classList.add("hidden");
    }, duration);
  }
});
