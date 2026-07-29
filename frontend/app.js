const API_BASE = window.METER_API_BASE || "http://localhost:8000";

const form = document.querySelector("#upload-form");
const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const previewGrid = document.querySelector("#preview-grid");
const summary = document.querySelector("#selection-summary");
const submitButton = document.querySelector("#submit-button");
const statusBox = document.querySelector("#upload-status");
const newResults = document.querySelector("#new-results");
const historyList = document.querySelector("#history-list");
const historyCount = document.querySelector("#history-count");
const resultTemplate = document.querySelector("#result-template");

let selectedFiles = [];

function endpoint(path) {
  return `${API_BASE}${path}`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat("vi-VN", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(value));
}

function setStatus(message, type = "") {
  statusBox.textContent = message;
  statusBox.className = `status ${type}`;
}

function statusClass(status) {
  return status === "ok" ? "" : "is-error";
}

function renderPreviews() {
  previewGrid.replaceChildren();
  summary.textContent = selectedFiles.length
    ? `Đã chọn ${selectedFiles.length} ảnh.`
    : "Chưa chọn ảnh.";
  submitButton.disabled = selectedFiles.length === 0;

  selectedFiles.forEach((file) => {
    const image = document.createElement("img");
    image.src = URL.createObjectURL(file);
    image.alt = `Xem trước ${file.name}`;
    image.onload = () => URL.revokeObjectURL(image.src);
    previewGrid.append(image);
  });
}

function addResultCard(record, prepend = true) {
  const card = resultTemplate.content.firstElementChild.cloneNode(true);
  const imageLink = card.querySelector(".result-image-link");
  const image = card.querySelector(".result-image");
  const badge = card.querySelector(".status-badge");
  const date = card.querySelector(".result-date");
  const reading = card.querySelector(".reading-value");
  const confidence = card.querySelector(".confidence-value");
  const correctionForm = card.querySelector(".correction-form");
  const correctionInput = card.querySelector(".correction-input");

  imageLink.href = endpoint(`/v1/readings/${record.id}/annotated-image`);
  image.src = imageLink.href;
  badge.textContent = record.status;
  if (statusClass(record.status)) badge.classList.add(statusClass(record.status));
  date.textContent = formatDate(record.detected_at);
  reading.textContent = record.corrected_reading || record.reading || "—";
  confidence.textContent = record.average_digit_confidence == null
    ? "Chưa có confidence chữ số"
    : `Confidence chữ số: ${(record.average_digit_confidence * 100).toFixed(1)}%`;
  correctionInput.value = record.corrected_reading || "";

  correctionForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = correctionForm.querySelector("button");
    button.disabled = true;
    try {
      const response = await fetch(endpoint(`/v1/readings/${record.id}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corrected_reading: correctionInput.value || null }),
      });
      if (!response.ok) throw new Error("Không thể lưu chỉ số đã sửa.");
      const updated = await response.json();
      reading.textContent = updated.corrected_reading || updated.reading || "—";
      setStatus("Đã lưu chỉ số đã xác nhận.", "success");
      loadHistory();
    } catch (error) {
      setStatus(error.message, "error");
    } finally {
      button.disabled = false;
    }
  });

  if (prepend) newResults.prepend(card);
  else newResults.append(card);
}

async function processFiles() {
  const digitConf = document.querySelector("#digit-conf").value;
  const rotateDegrees = document.querySelector("#rotate-degrees").value;
  newResults.classList.remove("empty-state");
  if (!newResults.children.length) newResults.replaceChildren();

  for (let index = 0; index < selectedFiles.length; index += 1) {
    const file = selectedFiles[index];
    setStatus(`Đang nhận diện ${index + 1}/${selectedFiles.length}: ${file.name}`);
    const payload = new FormData();
    payload.append("file", file);

    const response = await fetch(
      endpoint(`/v1/meter/read?digit_conf=${digitConf}&rotate_degrees=${rotateDegrees}`),
      { method: "POST", body: payload },
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Không thể xử lý ${file.name}.`);
    }
    addResultCard(await response.json());
  }

  setStatus(`Đã xử lý ${selectedFiles.length} ảnh và lưu vào PostgreSQL.`, "success");
  selectedFiles = [];
  input.value = "";
  renderPreviews();
  loadHistory();
}

async function loadHistory() {
  historyList.textContent = "Đang tải lịch sử...";
  try {
    const response = await fetch(endpoint("/v1/readings?limit=10"));
    if (!response.ok) throw new Error("Không thể tải lịch sử.");
    const records = await response.json();
    historyList.replaceChildren();
    historyCount.textContent = `${records.length} bản ghi gần nhất`;

    if (!records.length) {
      historyList.className = "history-list empty-state";
      historyList.textContent = "Chưa có bản ghi nào trong database.";
      return;
    }
    historyList.className = "history-list";
    records.forEach((record) => {
      const row = document.createElement("article");
      row.className = "history-row";
      const image = document.createElement("img");
      image.className = "history-thumb";
      image.src = endpoint(`/v1/readings/${record.id}/annotated-image`);
      image.alt = "Ảnh kết quả nhận diện";
      const text = document.createElement("div");
      const value = document.createElement("p");
      value.className = "history-reading";
      value.textContent = record.corrected_reading || record.reading || "—";
      const meta = document.createElement("p");
      meta.className = "history-meta";
      meta.textContent = `${formatDate(record.detected_at)} · ${record.image_width}×${record.image_height}`;
      text.append(value, meta);
      const badge = document.createElement("span");
badge.className = "status-badge";
      if (statusClass(record.status)) badge.classList.add(statusClass(record.status));
      badge.textContent = record.status;
      row.append(image, text, badge);
      historyList.append(row);
    });
  } catch (error) {
    historyList.className = "history-list empty-state";
    historyList.textContent = `${error.message} Kiểm tra API đã chạy ở port 8000.`;
  }
}

input.addEventListener("change", () => {
  selectedFiles = Array.from(input.files || []);
  renderPreviews();
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragover");
  });
});
dropZone.addEventListener("drop", (event) => {
  selectedFiles = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith("image/"));
  renderPreviews();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  try {
    await processFiles();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = selectedFiles.length === 0;
  }
});

document.querySelector("#refresh-button").addEventListener("click", loadHistory);
loadHistory();
