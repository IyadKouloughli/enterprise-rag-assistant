/**
 * NexusAI — Enterprise Knowledge & Operations Copilot Frontend Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const navButtons = document.querySelectorAll(".nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const roleSelect = document.getElementById("role-select");
  const aclRoleDisplay = document.getElementById("acl-role-display");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const chatForm = document.getElementById("chat-form");
  const queryInput = document.getElementById("query-input");
  const rerankToggle = document.getElementById("rerank-toggle");
  const messagesContainer = document.getElementById("messages-container");
  const welcomeHero = document.getElementById("welcome-hero");

  // Drawer Elements
  const sourceDrawer = document.getElementById("source-drawer");
  const drawerOverlay = document.getElementById("drawer-overlay");
  const drawerCloseBtn = document.getElementById("drawer-close-btn");
  const drawerCitationIdx = document.getElementById("drawer-citation-idx");
  const drawerTitle = document.getElementById("drawer-title");
  const drawerDept = document.getElementById("drawer-dept");
  const drawerPath = document.getElementById("drawer-path");
  const drawerScore = document.getElementById("drawer-score");
  const drawerRoles = document.getElementById("drawer-roles");
  const drawerText = document.getElementById("drawer-text");

  // State
  let conversationHistory = [];
  let currentSourcesMap = {}; // Maps citation_index -> source object

  // =========================================================================
  // 1. TAB NAVIGATION
  // =========================================================================
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      navButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(pane => pane.classList.remove("active"));

      btn.classList.add("active");
      const targetTab = btn.getAttribute("data-tab");
      document.getElementById(targetTab).classList.add("active");

      if (targetTab === "eval-tab") {
        loadEvaluationMetrics();
      } else if (targetTab === "docs-tab") {
        loadDocumentStats();
      }
    });
  });

  // =========================================================================
  // 2. ROLE SWITCHER
  // =========================================================================
  const roleLabels = {
    "hr": "Role: HR (Permissions: hr, general)",
    "engineer": "Role: Engineer (Permissions: engineer, support)",
    "finance": "Role: Finance (Permissions: finance, general)",
    "legal": "Role: Legal (Permissions: legal, general)",
    "support": "Role: Support (Permissions: support, engineer)",
    "engineer,manager": "Role: Lead (engineer + manager)",
    "admin": "Role: Superuser (ALL PERMISSIONS / Bypasses ACL)"
  };

  roleSelect.addEventListener("change", () => {
    const selected = roleSelect.value;
    aclRoleDisplay.textContent = roleLabels[selected] || `Role: ${selected}`;
  });

  // =========================================================================
  // 3. SUGGESTED QUERY CHIPS
  // =========================================================================
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      const r = chip.getAttribute("data-role");
      if (r) {
        roleSelect.value = r;
        aclRoleDisplay.textContent = roleLabels[r] || `Role: ${r}`;
      }
      queryInput.value = q;
      queryInput.focus();
      handleChatSubmit();
    });
  });

  // =========================================================================
  // 4. CHAT INTERACTION & CITATIONS
  // =========================================================================
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleChatSubmit();
  });

  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleChatSubmit();
    }
  });

  clearChatBtn.addEventListener("click", () => {
    conversationHistory = [];
    currentSourcesMap = {};
    messagesContainer.innerHTML = "";
    messagesContainer.appendChild(welcomeHero);
  });

  async function handleChatSubmit() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Remove welcome hero on first message
    if (welcomeHero && welcomeHero.parentNode === messagesContainer) {
      messagesContainer.removeChild(welcomeHero);
    }

    const userRole = roleSelect.value;
    const rerank = rerankToggle.checked;

    // 1. Render User Message
    renderUserMessage(query);
    queryInput.value = "";

    // 2. Render Bot Loading Bubble
    const botRow = renderBotLoading();
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    const streamStartTime = Date.now();
    let firstTokenSeconds = null;
    let accumulatedAnswer = "";
    let streamMeta = {
      sources: [],
      security_audit: null,
      citation_verification: null,
      latency: 0
    };

    const bubble = botRow.querySelector(".message-bubble");
    const textContainer = document.createElement("div");
    textContainer.className = "answer-text";
    bubble.innerHTML = "";
    bubble.appendChild(textContainer);

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonBody({
          query: query,
          role: userRole,
          history: conversationHistory,
          top_k: 5,
          rerank: rerank
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Server error");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data:")) {
            const dataStr = trimmed.replace(/^data:\s*/, "");
            if (!dataStr) continue;

            try {
              const payload = JSON.parse(dataStr);
              if (payload.type === "retrieval") {
                streamMeta.sources = payload.sources || [];
                streamMeta.security_audit = payload.security_audit;
                currentSourcesMap = {};
                streamMeta.sources.forEach((src, i) => {
                  currentSourcesMap[i + 1] = src;
                });
              } else if (payload.type === "token") {
                if (firstTokenSeconds === null) {
                  firstTokenSeconds = ((Date.now() - streamStartTime) / 1000).toFixed(2);
                }
                accumulatedAnswer += payload.content;
                // Fast streaming render
                textContainer.innerHTML = marked.parse(accumulatedAnswer);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
              } else if (payload.type === "done") {
                streamMeta.citation_verification = payload.citation_verification;
                streamMeta.latency = payload.latency || ((Date.now() - streamStartTime) / 1000).toFixed(2);
              } else if (payload.type === "error") {
                textContainer.innerHTML += `<div style="color: #f87171; margin-top: 8px;"><strong>Error:</strong> ${escapeHtml(payload.message)}</div>`;
              }
            } catch (jsonErr) {}
          }
        }
      }

      // Update history
      conversationHistory.push({ role: "user", content: query });
      conversationHistory.push({ role: "assistant", content: accumulatedAnswer });

      // Finalize and attach interactive citation links and metadata badges
      finalizeStreamedMessage(botRow, accumulatedAnswer, streamMeta, firstTokenSeconds);
    } catch (err) {
      bubble.innerHTML = `
        <div style="color: #f87171;">
          <strong>Error:</strong> ${escapeHtml(err.message)}
        </div>
      `;
    }

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function finalizeStreamedMessage(row, rawAnswer, meta, ttft) {
    const bubble = row.querySelector(".message-bubble");

    // Replace [1], [2] with clickable badge buttons
    let rendered = marked.parse(rawAnswer);
    rendered = rendered.replace(/\[(\d+)\]/g, (match, idx) => {
      return `<button class="citation-badge" data-citation="${idx}">[${idx}]</button>`;
    });

    // Verification Badge
    let verifPill = "";
    if (meta.citation_verification) {
      const cv = meta.citation_verification;
      const badgeClass = cv.status === "VERIFIED" ? "green" : (cv.status === "WARNING" ? "yellow" : "red");
      verifPill = `
        <span class="meta-pill" title="${cv.status_message}">
          🛡️ Grounding: <strong style="color: #34d399;">${cv.status}</strong> (${Math.round(cv.sentence_citation_coverage * 100)}% cited)
        </span>
      `;
    }

    // Security Audit Pill
    let auditPill = "";
    if (meta.security_audit) {
      const sa = meta.security_audit;
      auditPill = `
        <span class="meta-pill">
          🔒 ACL: ${sa.candidates_passed_acl} passed | ${sa.candidates_blocked_by_acl} blocked
        </span>
      `;
    }

    // Latency & TTFT Pill
    const ttftText = ttft ? `TTFT: ${ttft}s | ` : "";
    const latencyPill = `
      <span class="meta-pill">
        ⚡ ${ttftText}Total: ${meta.latency}s
      </span>
    `;

    // Sources Chips List
    let sourcesChipsHtml = "";
    if (meta.sources && meta.sources.length > 0) {
      sourcesChipsHtml = `
        <div style="margin-top: 10px; font-size: 0.78rem; color: var(--text-dim); font-weight: 600;">Cited Sources:</div>
        <div class="sources-chips-list">
          ${meta.sources.map((s, i) => `
            <button class="source-item-btn" data-citation="${i + 1}">
              <strong>[${i + 1}]</strong> ${escapeHtml(s.title || 'Document')} (${s.department || 'general'})
            </button>
          `).join("")}
        </div>
      `;
    }

    bubble.innerHTML = `
      <div class="answer-text">${rendered}</div>
      ${sourcesChipsHtml}
      <div class="bot-meta-footer">
        ${verifPill}
        ${auditPill}
        ${latencyPill}
      </div>
    `;

    // Attach click listeners to citation badges & source buttons
    bubble.querySelectorAll("[data-citation]").forEach(btn => {
      btn.addEventListener("click", () => {
        const citationNum = parseInt(btn.getAttribute("data-citation"));
        openSourceDrawer(citationNum);
      });
    });
  }

  function jsonBody(obj) {
    return JSON.stringify(obj);
  }

  function renderUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.innerHTML = `
      <div class="message-bubble user-bubble">${escapeHtml(text)}</div>
      <div class="avatar user-avatar">U</div>
    `;
    messagesContainer.appendChild(row);
  }

  function renderBotLoading() {
    const row = document.createElement("div");
    row.className = "message-row bot-row";
    row.innerHTML = `
      <div class="avatar bot-avatar">AI</div>
      <div class="message-bubble bot-bubble">
        <div style="color: var(--text-muted); font-style: italic;">
          Retrieving knowledge & verifying grounding...
        </div>
      </div>
    `;
    messagesContainer.appendChild(row);
    return row;
  }

  function updateBotMessage(row, data) {
    const bubble = row.querySelector(".message-bubble");
    
    // Parse markdown and replace [1], [2] with interactive clickable badge elements
    let renderedAnswer = marked.parse(data.answer);
    renderedAnswer = renderedAnswer.replace(/\[(\d+)\]/g, (match, idx) => {
      return `<button class="citation-badge" data-citation="${idx}">[${idx}]</button>`;
    });

    // Verification Badge
    let verifPill = "";
    if (data.citation_verification) {
      const cv = data.citation_verification;
      const badgeClass = cv.status === "VERIFIED" ? "green" : (cv.status === "WARNING" ? "yellow" : "red");
      verifPill = `
        <span class="meta-pill" title="${cv.status_message}">
          🛡️ Grounding: <strong style="color: #34d399;">${cv.status}</strong> (${Math.round(cv.sentence_citation_coverage * 100)}% cited)
        </span>
      `;
    }

    // Security Audit Pill
    let auditPill = "";
    if (data.security_audit) {
      const sa = data.security_audit;
      auditPill = `
        <span class="meta-pill">
          🔒 ACL: ${sa.candidates_passed_acl} passed | ${sa.candidates_blocked_by_acl} blocked
        </span>
      `;
    }

    // Latency Pill
    const latencyPill = `
      <span class="meta-pill">
        ⚡ ${data.latency_seconds}s
      </span>
    `;

    // Sources Chips List
    let sourcesChipsHtml = "";
    if (data.sources && data.sources.length > 0) {
      sourcesChipsHtml = `
        <div style="margin-top: 10px; font-size: 0.78rem; color: var(--text-dim); font-weight: 600;">Cited Sources:</div>
        <div class="sources-chips-list">
          ${data.sources.map((s, i) => `
            <button class="source-item-btn" data-citation="${i + 1}">
              <strong>[${i + 1}]</strong> ${escapeHtml(s.title || 'Document')} (${s.department || 'general'})
            </button>
          `).join("")}
        </div>
      `;
    }

    bubble.innerHTML = `
      <div class="answer-text">${renderedAnswer}</div>
      ${sourcesChipsHtml}
      <div class="bot-meta-footer">
        ${verifPill}
        ${auditPill}
        ${latencyPill}
      </div>
    `;

    // Attach click listeners to citation badges & source buttons
    bubble.querySelectorAll("[data-citation]").forEach(btn => {
      btn.addEventListener("click", () => {
        const citationNum = parseInt(btn.getAttribute("data-citation"));
        openSourceDrawer(citationNum);
      });
    });
  }

  // =========================================================================
  // 5. SOURCE INSPECTOR DRAWER
  // =========================================================================
  function openSourceDrawer(idx) {
    const src = currentSourcesMap[idx];
    if (!src) return;

    drawerCitationIdx.textContent = `[${idx}]`;
    drawerTitle.textContent = src.title || "Document Chunk";
    drawerDept.textContent = (src.department || "general").toUpperCase();
    drawerPath.textContent = src.source_path || "N/A";
    
    if (src.rerank_score !== undefined) {
      drawerScore.textContent = `${src.rerank_score.toFixed(4)} (Cross-Encoder)`;
    } else if (src.fused_score !== undefined) {
      drawerScore.textContent = `${src.fused_score.toFixed(4)} (RRF)`;
    } else {
      drawerScore.textContent = "N/A";
    }

    drawerRoles.textContent = JSON.stringify(src.allowed_roles || ["all"]);
    drawerText.textContent = src.text || "";

    sourceDrawer.classList.add("open");
    drawerOverlay.classList.add("open");
  }

  function closeSourceDrawer() {
    sourceDrawer.classList.remove("open");
    drawerOverlay.classList.remove("open");
  }

  drawerCloseBtn.addEventListener("click", closeSourceDrawer);
  drawerOverlay.addEventListener("click", closeSourceDrawer);

  // =========================================================================
  // 6. EVALUATION METRICS TAB
  // =========================================================================
  async function loadEvaluationMetrics() {
    try {
      const res = await fetch("/api/evaluation");
      if (!res.ok) return;
      const evalData = await res.json();

      const rerank = evalData.retrieval_reranked || {};
      const base = evalData.retrieval_baseline || {};
      const gen = evalData.generation_metrics || {};

      if (rerank.recall_at_k !== undefined) {
        document.getElementById("eval-recall").textContent = `${(rerank.recall_at_k * 100).toFixed(1)}%`;
      }
      if (rerank.precision_at_k !== undefined) {
        document.getElementById("eval-precision").textContent = `${(rerank.precision_at_k * 100).toFixed(1)}%`;
      }
      if (rerank.mrr !== undefined) {
        document.getElementById("eval-mrr").textContent = rerank.mrr.toFixed(3);
      }
      if (rerank.ndcg_at_k !== undefined) {
        document.getElementById("eval-ndcg").textContent = rerank.ndcg_at_k.toFixed(3);
      }
      if (gen.faithfulness !== undefined) {
        document.getElementById("eval-faithfulness").textContent = `${(gen.faithfulness * 100).toFixed(1)}%`;
      }
      if (gen.citation_accuracy !== undefined) {
        document.getElementById("eval-citations").textContent = `${(gen.citation_accuracy * 100).toFixed(1)}%`;
      }

      // Populate Table
      const tbody = document.getElementById("eval-table-body");
      tbody.innerHTML = "";
      (evalData.per_test_details || []).forEach(item => {
        const rm = item.reranked_retrieval || {};
        const gm = item.generation || {};
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${item.id}</code></td>
          <td>${escapeHtml(item.category)}</td>
          <td><span class="chip-badge eng">${item.role}</span></td>
          <td><strong>${(rm.recall_at_k * 100).toFixed(0)}%</strong></td>
          <td>${rm.mrr.toFixed(2)}</td>
          <td><span style="color: #34d399;">${(gm.faithfulness * 100).toFixed(0)}%</span></td>
          <td><code>${item.citations_summary || 'VERIFIED'}</code></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.warn("Could not load evaluation metrics:", e);
    }
  }

  // =========================================================================
  // 7. KNOWLEDGE BASE STATS TAB
  // =========================================================================
  async function loadDocumentStats() {
    try {
      const res = await fetch("/api/documents");
      if (!res.ok) return;
      const stats = await res.json();

      document.getElementById("total-chunks-val").textContent = stats.total_chunks.toLocaleString();

      const deptContainer = document.getElementById("departments-container");
      deptContainer.innerHTML = "";
      Object.entries(stats.departments || {}).forEach(([dept, count]) => {
        const card = document.createElement("div");
        card.className = "dept-card";
        card.innerHTML = `
          <div class="dept-card-header">${dept}</div>
          <div class="dept-card-count">${count.toLocaleString()} <span style="font-size: 0.75rem; color: var(--text-dim); font-weight: normal;">chunks</span></div>
        `;
        deptContainer.appendChild(card);
      });

      const docsBody = document.getElementById("sample-docs-body");
      docsBody.innerHTML = "";
      (stats.sample_documents || []).forEach(doc => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${escapeHtml(doc.title || 'Untitled')}</strong></td>
          <td><span class="chip-badge hr">${doc.department}</span></td>
          <td style="font-family: var(--font-mono); font-size: 0.75rem; color: #93c5fd;">${doc.source_path}</td>
          <td><span style="font-size: 0.75rem; color: var(--text-dim);">${JSON.stringify(doc.allowed_roles || [])}</span></td>
        `;
        docsBody.appendChild(tr);
      });
    } catch (e) {
      console.warn("Could not load document stats:", e);
    }
  }

  // Check initial system health
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        document.getElementById("active-model-tag").textContent = data.model_name.split("/").pop();
      }
    } catch (e) {}
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  checkHealth();
});
