
"use strict";

// ============================================
// Application State
// ============================================

const STORAGE_KEY = "agentic_rag_current_thread";

let currentThreadId = localStorage.getItem(STORAGE_KEY) || null;
let isSending = false;

const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const chatMessages = document.getElementById("chat-messages");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");

const fileUpload = document.getElementById("file-upload");
const uploadStatus = document.getElementById("upload-status");
const documentsList = document.getElementById("documents-list");
const chatHistory = document.getElementById("chat-history");


// ============================================
// API Helpers
// ============================================

async function readResponse(response) {
    const contentType = response.headers.get("content-type") || "";

    if (!contentType.includes("application/json")) {
        const body = await response.text();

        throw new Error(
            `The server returned an unexpected response. ${body.slice(0, 150)}`
        );
    }

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Something went wrong.");
    }

    return data;
}


async function apiRequest(url, options = {}) {
    const response = await fetch(url, options);
    return readResponse(response);
}


// ============================================
// Safe Markdown Renderer
// ============================================

function appendInlineMarkdown(parent, text) {
    const pattern =
        /(`[^`]+`|\*\*.+?\*\*|__.+?__|\*[^*]+\*|_[^_]+_|<br\s*\/?>|\[[^\]]+\]\(https?:\/\/[^)\s]+\)|https?:\/\/[^\s<]+)/g;

    let lastIndex = 0;
    let match;

    while ((match = pattern.exec(text)) !== null) {
        if (match.index > lastIndex) {
            parent.appendChild(
                document.createTextNode(text.slice(lastIndex, match.index))
            );
        }

        const token = match[0];
        let element = null;

        if (/^<br\s*\/?>$/i.test(token)) {
            element = document.createElement("br");

        } else if (token.startsWith("`")) {
            element = document.createElement("code");
            element.textContent = token.slice(1, -1);

        } else if (
            token.startsWith("**") ||
            token.startsWith("__")
        ) {
            element = document.createElement("strong");
            element.textContent = token.slice(2, -2);

        } else if (
            token.startsWith("*") ||
            token.startsWith("_")
        ) {
            element = document.createElement("em");
            element.textContent = token.slice(1, -1);

        } else if (token.startsWith("[")) {
            const linkMatch = token.match(
                /^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/
            );

            if (linkMatch) {
                element = document.createElement("a");
                element.textContent = linkMatch[1];
                element.href = linkMatch[2];
                element.target = "_blank";
                element.rel = "noopener noreferrer";
            }

        } else if (/^https?:\/\//i.test(token)) {
            element = document.createElement("a");
            element.textContent = token;
            element.href = token;
            element.target = "_blank";
            element.rel = "noopener noreferrer";
        }

        parent.appendChild(element || document.createTextNode(token));
        lastIndex = pattern.lastIndex;
    }

    if (lastIndex < text.length) {
        parent.appendChild(
            document.createTextNode(text.slice(lastIndex))
        );
    }
}


function parseTableRow(line) {
    let value = line.trim();

    if (value.startsWith("|")) value = value.slice(1);
    if (value.endsWith("|")) value = value.slice(0, -1);

    return value.split("|").map(cell => cell.trim());
}


function isTableSeparator(line) {
    const cells = parseTableRow(line);

    return (
        cells.length > 0 &&
        cells.every(cell => /^:?-{3,}:?$/.test(cell))
    );
}


function renderTable(lines) {
    const wrapper = document.createElement("div");
    wrapper.className = "markdown-table-wrapper";
    wrapper.style.overflowX = "auto";

    const table = document.createElement("table");
    table.className = "markdown-table";
    table.style.borderCollapse = "collapse";
    table.style.width = "100%";

    const headers = parseTableRow(lines[0]);
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    headers.forEach(value => {
        const th = document.createElement("th");
        th.textContent = value;
        th.style.padding = "10px";
        th.style.textAlign = "left";
        th.style.borderBottom = "2px solid #999";
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    for (let i = 2; i < lines.length; i++) {
        const values = parseTableRow(lines[i]);
        const row = document.createElement("tr");

        headers.forEach((_, index) => {
            const td = document.createElement("td");
            td.style.padding = "10px";
            td.style.borderBottom = "1px solid #ddd";

            appendInlineMarkdown(td, values[index] || "");
            row.appendChild(td);
        });

        tbody.appendChild(row);
    }

    table.appendChild(tbody);
    wrapper.appendChild(table);

    return wrapper;
}


function renderMarkdown(content) {
    const container = document.createElement("div");
    container.className = "markdown-body";

    const normalized = String(content || "")
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n")
        .replace(/<br\s*\/?>/gi, "\n");

    const lines = normalized.split("\n");
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];
        const trimmed = line.trim();

        if (!trimmed) {
            i++;
            continue;
        }

        // Fenced code blocks
        if (trimmed.startsWith("```")) {
            const codeLines = [];
            i++;

            while (
                i < lines.length &&
                !lines[i].trim().startsWith("```")
            ) {
                codeLines.push(lines[i]);
                i++;
            }

            if (i < lines.length) i++;

            const pre = document.createElement("pre");
            const code = document.createElement("code");

            code.textContent = codeLines.join("\n");
            pre.appendChild(code);
            container.appendChild(pre);
            continue;
        }

        // Markdown tables
        if (
            line.includes("|") &&
            i + 1 < lines.length &&
            isTableSeparator(lines[i + 1])
        ) {
            const tableLines = [line];
            i += 2;

            while (
                i < lines.length &&
                lines[i].trim() &&
                lines[i].includes("|")
            ) {
                tableLines.push(lines[i]);
                i++;
            }

            container.appendChild(renderTable(tableLines));
            continue;
        }

        // Headings
        const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);

        if (headingMatch) {
            const level = headingMatch[1].length;
            const heading = document.createElement(`h${level}`);

            appendInlineMarkdown(heading, headingMatch[2]);
            container.appendChild(heading);
            i++;
            continue;
        }

        // Blockquotes
        if (trimmed.startsWith(">")) {
            const quote = document.createElement("blockquote");

            while (
                i < lines.length &&
                lines[i].trim().startsWith(">")
            ) {
                const quoteText = lines[i]
                    .trim()
                    .replace(/^>\s?/, "");

                const paragraph = document.createElement("p");
                appendInlineMarkdown(paragraph, quoteText);
                quote.appendChild(paragraph);
                i++;
            }

            container.appendChild(quote);
            continue;
        }

        // Unordered lists
        if (/^[-*+]\s+/.test(trimmed)) {
            const list = document.createElement("ul");

            while (
                i < lines.length &&
                /^[-*+]\s+/.test(lines[i].trim())
            ) {
                const item = document.createElement("li");
                const value = lines[i]
                    .trim()
                    .replace(/^[-*+]\s+/, "");

                appendInlineMarkdown(item, value);
                list.appendChild(item);
                i++;
            }

            container.appendChild(list);
            continue;
        }

        // Ordered lists
        if (/^\d+[.)]\s+/.test(trimmed)) {
            const list = document.createElement("ol");

            while (
                i < lines.length &&
                /^\d+[.)]\s+/.test(lines[i].trim())
            ) {
                const item = document.createElement("li");
                const value = lines[i]
                    .trim()
                    .replace(/^\d+[.)]\s+/, "");

                appendInlineMarkdown(item, value);
                list.appendChild(item);
                i++;
            }

            container.appendChild(list);
            continue;
        }

        // Regular paragraphs
        const paragraph = document.createElement("p");
        const paragraphLines = [];

        while (
            i < lines.length &&
            lines[i].trim() &&
            !lines[i].trim().startsWith("```") &&
            !/^#{1,6}\s+/.test(lines[i].trim()) &&
            !lines[i].trim().startsWith(">") &&
            !/^[-*+]\s+/.test(lines[i].trim()) &&
            !/^\d+[.)]\s+/.test(lines[i].trim()) &&
            !(
                lines[i].includes("|") &&
                i + 1 < lines.length &&
                isTableSeparator(lines[i + 1])
            )
        ) {
            paragraphLines.push(lines[i].trim());
            i++;
        }

        appendInlineMarkdown(
            paragraph,
            paragraphLines.join(" ")
        );

        container.appendChild(paragraph);
    }

    return container;
}


// ============================================
// Chat UI
// ============================================

function showWelcome() {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <h2>Welcome to Agentic RAG</h2>
            <p>Ask questions about your uploaded documents or the web.</p>
        </div>
    `;
}


function addMessage(role, content) {
    const welcome = chatMessages.querySelector(".welcome-message");

    if (welcome) welcome.remove();

    const messageElement = document.createElement("div");
    messageElement.className = `message ${role}-message`;

    const contentElement = document.createElement("div");
    contentElement.className = "message-content";

    if (role === "assistant") {
        contentElement.appendChild(renderMarkdown(content || ""));
    } else {
        contentElement.textContent = content || "";
    }

    messageElement.appendChild(contentElement);
    chatMessages.appendChild(messageElement);

    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageElement;
}


function addLoadingMessage() {
    const element = addMessage("assistant", "Thinking...");
    element.classList.add("loading-message");
    return element;
}


// ============================================
// Agent Trace
// ============================================

function showTrace(trace, parentElement = null) {
    if (!Array.isArray(trace) || trace.length === 0) return;

    const container = document.createElement("details");
    container.className = "trace-container";

    const summary = document.createElement("summary");
    summary.textContent = "View Agent Trace";

    const traceContent = document.createElement("div");
    traceContent.className = "trace-content";

    trace.forEach(item => {
        const toolElement = document.createElement("div");
        toolElement.className = "trace-item";

        const toolName = document.createElement("strong");
        toolName.textContent = item.tool || "Tool";

        const output = document.createElement("pre");
        output.textContent = item.output || "";

        toolElement.appendChild(toolName);
        toolElement.appendChild(output);
        traceContent.appendChild(toolElement);
    });

    container.appendChild(summary);
    container.appendChild(traceContent);

    if (parentElement) {
        parentElement.appendChild(container);
    } else {
        chatMessages.appendChild(container);
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// ============================================
// Sources
// ============================================

function showSources(sources, parentElement) {
    if (!Array.isArray(sources) || sources.length === 0) return;

    const container = document.createElement("div");
    container.className = "sources-container";

    const heading = document.createElement("strong");
    heading.textContent = "Sources";
    container.appendChild(heading);

    const list = document.createElement("ul");

    sources.forEach((source, index) => {
        const item = document.createElement("li");

        if (source.type === "web" && source.url) {
            const link = document.createElement("a");

            link.href = source.url;
            link.textContent =
                `[${index + 1}] ${source.title || source.url}`;

            link.target = "_blank";
            link.rel = "noopener noreferrer";

            item.appendChild(link);

            if (source.description) {
                const description = document.createElement("p");
                description.textContent = source.description;
                item.appendChild(description);
            }

        } else if (
            source.type === "document" ||
            source.source
        ) {
            item.textContent =
                `${source.source || "Document"}` +
                (source.page ? ` — Page ${source.page}` : "");

        } else {
            return;
        }

        list.appendChild(item);
    });

    if (list.children.length === 0) return;

    container.appendChild(list);
    parentElement.appendChild(container);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// ============================================
// Chat History
// ============================================

async function loadConversations() {
    try {
        const data = await apiRequest("/api/conversations");

        const conversations = Array.isArray(data)
            ? data
            : (data.conversations || []);

        renderConversations(conversations);

        return conversations;

    } catch (error) {
        console.error("Could not load conversations:", error);

        if (chatHistory) {
            chatHistory.innerHTML = `
                <p class="error-message">
                    Could not load chat history.
                </p>
            `;
        }

        return [];
    }
}


function renderConversations(conversations) {
    if (!chatHistory) return;

    chatHistory.innerHTML = "";

    if (!conversations || conversations.length === 0) {
        chatHistory.innerHTML = `
            <p class="empty-history">No previous conversations.</p>
        `;
        return;
    }

    conversations.forEach(conversation => {
        const threadId = conversation.thread_id;

        const row = document.createElement("div");
        row.className = "history-row";

        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.gap = "6px";
        row.style.width = "100%";
        row.style.marginBottom = "6px";

        const button = document.createElement("button");
        button.type = "button";
        button.className = "history-item";
        button.textContent = conversation.title || "New Chat";

        button.style.flex = "1";
        button.style.minWidth = "0";
        button.style.textAlign = "left";

        if (threadId === currentThreadId) {
            button.classList.add("active");
        }

        button.addEventListener("click", () => {
            openConversation(threadId);
        });

        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "delete-chat-btn";
        deleteButton.textContent = "×";
        deleteButton.title = "Delete conversation";
        deleteButton.setAttribute(
            "aria-label",
            "Delete conversation"
        );

        deleteButton.style.flex = "0 0 32px";
        deleteButton.style.width = "32px";
        deleteButton.style.height = "32px";
        deleteButton.style.padding = "0";
        deleteButton.style.cursor = "pointer";
        deleteButton.style.border = "none";
        deleteButton.style.borderRadius = "6px";
        deleteButton.style.fontSize = "21px";
        deleteButton.style.background = "transparent";

        deleteButton.addEventListener("click", () => {
            deleteConversation(threadId);
        });

        row.appendChild(button);
        row.appendChild(deleteButton);

        chatHistory.appendChild(row);
    });
}


async function openConversation(threadId) {
    if (!threadId) return;

    try {
        const data = await apiRequest(
            `/api/conversation/${encodeURIComponent(threadId)}`
        );

        currentThreadId = threadId;
        localStorage.setItem(STORAGE_KEY, threadId);

        chatMessages.innerHTML = "";

        const messages = data.messages || [];

        if (messages.length === 0) {
            showWelcome();

        } else {
            messages.forEach(message => {
                const element = addMessage(
                    message.role,
                    message.content
                );

                if (message.role === "assistant") {
                    showSources(message.sources || [], element);
                    showTrace(message.trace || [], element);
                }
            });
        }

        await loadConversations();

    } catch (error) {
        console.error("Could not open conversation:", error);
        alert(`Could not open conversation: ${error.message}`);
    }
}


// ============================================
// Delete Conversation
// ============================================

async function deleteConversation(threadId) {
    if (!threadId) return;

    if (isSending && threadId === currentThreadId) {
        alert("Please wait until the current response finishes.");
        return;
    }

    const confirmed = confirm(
        "Are you sure you want to permanently delete this conversation?"
    );

    if (!confirmed) return;

    try {
        await apiRequest(
            `/api/conversation/${encodeURIComponent(threadId)}`,
            {
                method: "DELETE",
            }
        );

        if (threadId === currentThreadId) {
            currentThreadId = null;
            localStorage.removeItem(STORAGE_KEY);

            showWelcome();

            await createNewChat();

        } else {
            await loadConversations();
        }

    } catch (error) {
        console.error("Could not delete conversation:", error);
        alert(`Could not delete conversation: ${error.message}`);
    }
}


// ============================================
// New Chat
// ============================================

async function createNewChat() {
    if (isSending) return;

    try {
        const data = await apiRequest("/api/conversations", {
            method: "POST",
        });

        currentThreadId = data.thread_id;
        localStorage.setItem(STORAGE_KEY, currentThreadId);

        showWelcome();

        await loadConversations();

        messageInput.focus();

    } catch (error) {
        console.error("Could not create a new chat:", error);
        alert(`Could not create a new chat: ${error.message}`);
    }
}


if (newChatBtn) {
    newChatBtn.addEventListener("click", createNewChat);
}


// ============================================
// Send Message
// ============================================

chatForm.addEventListener("submit", async event => {
    event.preventDefault();

    if (isSending) return;

    const message = messageInput.value.trim();

    if (!message) return;

    if (!currentThreadId) {
        try {
            const data = await apiRequest("/api/conversations", {
                method: "POST",
            });

            currentThreadId = data.thread_id;
            localStorage.setItem(STORAGE_KEY, currentThreadId);

        } catch (error) {
            alert(`Could not create conversation: ${error.message}`);
            return;
        }
    }

    isSending = true;

    const submittedThreadId = currentThreadId;

    messageInput.value = "";
    sendBtn.disabled = true;

    addMessage("user", message);

    const loadingElement = addLoadingMessage();

    try {
        const data = await apiRequest("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message,
                thread_id: submittedThreadId,
            }),
        });

        loadingElement.remove();

        if (currentThreadId === submittedThreadId) {
            const assistantElement = addMessage(
                "assistant",
                data.answer || "No answer was returned."
            );

            showSources(data.sources || [], assistantElement);
            showTrace(data.trace || [], assistantElement);
        }

        await loadConversations();

    } catch (error) {
        loadingElement.remove();

        if (currentThreadId === submittedThreadId) {
            addMessage(
                "assistant",
                `Error: ${error.message}`
            );
        }

        console.error("Chat request failed:", error);

    } finally {
        isSending = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
});


// ============================================
// Document Upload
// ============================================

if (fileUpload) {
    fileUpload.addEventListener("change", async () => {
        const file = fileUpload.files[0];

        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        if (uploadStatus) {
            uploadStatus.textContent = `Uploading ${file.name}...`;
        }

        fileUpload.disabled = true;

        try {
            const data = await apiRequest("/api/upload", {
                method: "POST",
                body: formData,
            });

            if (uploadStatus) {
                uploadStatus.textContent =
                    `${data.filename} uploaded successfully. ` +
                    `${data.chunks} chunks indexed.`;
            }

            await loadDocuments();

        } catch (error) {
            if (uploadStatus) {
                uploadStatus.textContent =
                    `Upload failed: ${error.message}`;
            }

            console.error("Upload failed:", error);

        } finally {
            fileUpload.disabled = false;
            fileUpload.value = "";
        }
    });
}


// ============================================
// Documents List
// ============================================

function renderDocuments(documents) {
    if (!documentsList) return;

    documentsList.innerHTML = "";

    if (!documents || documents.length === 0) {
        documentsList.innerHTML = `
            <p class="empty-documents">No documents uploaded.</p>
        `;
        return;
    }

    documents.forEach(document => {
        const filename = typeof document === "string"
            ? document
            : (document.source || document.filename || document.name);

        if (!filename) return;

        const item = documentElement(filename);
        documentsList.appendChild(item);
    });
}


function documentElement(filename) {
    const item = document.createElement("div");
    item.className = "document-item";

    const name = document.createElement("span");
    name.className = "document-name";
    name.textContent = filename;

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-document-btn";
    deleteButton.textContent = "×";
    deleteButton.title = "Delete document";

    deleteButton.addEventListener("click", () => {
        deleteDocument(filename);
    });

    item.appendChild(name);
    item.appendChild(deleteButton);

    return item;
}


async function loadDocuments() {
    try {
        const data = await apiRequest("/api/documents");
        renderDocuments(data.documents || []);

    } catch (error) {
        console.error("Could not load documents:", error);

        if (documentsList) {
            documentsList.innerHTML = `
                <p class="error-message">
                    Could not load documents.
                </p>
            `;
        }
    }
}


async function deleteDocument(filename) {
    const confirmed = confirm(
        `Are you sure you want to delete ${filename}?`
    );

    if (!confirmed) return;

    try {
        await apiRequest(
            `/api/documents/${encodeURIComponent(filename)}`,
            {
                method: "DELETE",
            }
        );

        await loadDocuments();

    } catch (error) {
        alert(`Could not delete document: ${error.message}`);
    }
}


// ============================================
// Application Initialization
// ============================================

async function initializeApp() {
    await loadDocuments();

    const conversations = await loadConversations();

    if (currentThreadId) {
        const savedConversation = conversations.find(
            conversation =>
                conversation.thread_id === currentThreadId
        );

        if (savedConversation) {
            await openConversation(currentThreadId);
            return;
        }
    }

    if (conversations.length > 0) {
        await openConversation(conversations[0].thread_id);
        return;
    }

    await createNewChat();
}


// ============================================
// Keyboard Handling
// ============================================

messageInput.addEventListener("keydown", event => {
    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {
        event.preventDefault();

        if (!isSending) {
            chatForm.requestSubmit();
        }
    }
});


// ============================================
// Start Application
// ============================================

initializeApp();
