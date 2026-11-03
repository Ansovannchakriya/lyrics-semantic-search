const queryInput = document.getElementById("query-input");
const searchBtn = document.getElementById("search-btn");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");

async function runSearch() {
  const query = queryInput.value.trim();
  errorEl.classList.add("hidden");
  resultsEl.innerHTML = "";

  if (!query) {
    errorEl.textContent = "Type a description of the song you're looking for first.";
    errorEl.classList.remove("hidden");
    return;
  }

  loadingEl.classList.remove("hidden");
  searchBtn.disabled = true;

  try {
    const response = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, top_k: 10 }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Something went wrong.");
    }

    renderResults(data.results);
  } catch (err) {
    errorEl.textContent = err.message || "Search failed. Please try again.";
    errorEl.classList.remove("hidden");
  } finally {
    loadingEl.classList.add("hidden");
    searchBtn.disabled = false;
  }
}

function renderResults(results) {
  resultsEl.innerHTML = "";

  if (!results || results.length === 0) {
    resultsEl.innerHTML = '<li class="empty-state">No strong matches found. Try describing the mood differently, or lower minimum_score in configs/config.yaml.</li>';
    return;
  }

  for (const r of results) {
    const li = document.createElement("li");
    li.className = "result-card";
    const sc = r.scores || {};
    const matched = r.matched_lyrics || r.lyrics_excerpt;
    li.innerHTML = `
      <div class="result-top">
        <div class="result-title">${escapeHtml(r.song)} <span class="result-artist">— ${escapeHtml(r.artist)}</span></div>
        <div class="result-score" title="raw cosine similarity ${r.similarity_score}">#${r.rank} · ${headline(r)}</div>
      </div>
      <p class="result-excerpt"><span class="matched-label">best matching lines</span> "${escapeHtml(matched)}"</p>
      ${r.scores ? `<div class="score-breakdown">
        <span title="blend of the three cosines below">cosine ${r.similarity_score}</span>
        ${r.relevance != null && r.match_percent != null ? `<span title="cosine rescaled for display">match ${r.match_percent}%</span>` : ""}
        <span title="best matching lyric chunk">lyrics chunk ${sc.best_chunk}</span>
        <span title="whole song (all chunks averaged)">whole song ${sc.whole_song}</span>
        <span title="song title">title ${sc.title}</span>
      </div>` : ""}
      <div class="result-link"><a href="${escapeAttr(fullLyricsUrl(r.link))}" target="_blank" rel="noopener">view lyrics source ↗</a></div>
    `;
    resultsEl.appendChild(li);
  }
}

// Main score on each card: re-ranker relevance (0-1) when available,
// otherwise the display "match %", otherwise the raw cosine.
function headline(r) {
  if (r.relevance != null) return `relevance ${r.relevance.toFixed(2)}`;
  if (r.match_percent != null) return `match ${r.match_percent}%`;
  return `similarity ${r.similarity_score}`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function escapeAttr(str) {
  return (str ?? "").replace(/"/g, "&quot;");
}

// The dataset's "link" column stores relative paths from the site the
// lyrics were originally sourced from (lyricsfreak.com), not full URLs.
function fullLyricsUrl(link) {
  if (!link) return "#";
  if (link.startsWith("http")) return link;
  return "https://www.lyricsfreak.com" + link;
}

searchBtn.addEventListener("click", runSearch);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});

document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    queryInput.value = chip.textContent;
    runSearch();
  });
});
