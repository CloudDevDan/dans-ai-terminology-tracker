const fs = require('fs');
const path = './docs/index.html'; // Path to your Markmap export

let html;
try {
  html = fs.readFileSync(path, 'utf8');
} catch (err) {
  console.error(`❌ Error reading ${path}:`, err.message);
  process.exit(1);
}

// ✅ Inject favicon and manifest links if missing
if (!html.includes('apple-touch-icon')) {
  html = html.replace('</head>', `<link rel="icon" type="image/png" href="favicon/favicon.png">
<link rel="apple-touch-icon" sizes="180x180" href="favicon/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="favicon/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="favicon/favicon-16x16.png">
<link rel="manifest" href="favicon/site.webmanifest">
</head>`);
}

// ✅ Update <title>
html = html.replace(
  /<title>.*?<\/title>/,
  '<title>Dan’s AI Terminology Tracker</title>'
);

// ✅ Update #mindmap style with dark background + white text
html = html.replace(
  /#mindmap\s*\{[^}]*\}/,
  match => {
    let updated = match;
    if (!updated.includes('color')) {
      updated = updated.replace('}', '  color: white;\n}');
    }
    if (!updated.includes('background-color')) {
      updated = updated.replace('}', '  background-color: #1a1b26;\n}');
    }
    return updated;
  }
);

// ✅ Update <body> CSS block to include dark background
html = html.replace(
  /body\s*\{[^}]*\}/,
  match => {
    if (!match.includes('background-color')) {
      return match.replace('}', '  background-color: #1a1b26;\n}');
    }
    return match;
  }
);

// ✅ Replace <body> tag with a clean one (prevents rendering bugs)
html = html.replace(
  /<body[^>]*>/,
  '<body style="background-color: #1a1b26;">'
);


// ✅ Inject styled header block with external toggle button
const headerHTML = `
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
  body {
    font-family: 'Inter', sans-serif;
  }
  #toggleHeaderButton {
    position: fixed;
    top: 1rem;
    right: 1rem;
    background-color: #333;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    border-radius: 8px;
    cursor: pointer;
    z-index: 1000;
  }
  #toggleHeaderButton:hover {
    background-color: #555;
  }
</style>
<button id="toggleHeaderButton">Hide Header</button>
<div id="headerWrapper">
  <header style="
    padding: 2rem 1rem;
    background-color: #0d0d0d;
    color: white;
    text-align: center;
    border-bottom: 1px solid #222;
  ">
    <h1 style="font-size: 2.25rem; margin: 0; font-weight: 600;">
      🧠 Dan’s AI Terminology Tracker
    </h1>
    <p style="margin-top: 0.75rem; font-size: 1.05rem; color: #aaa; max-width: 800px; margin-left: auto; margin-right: auto;">
      A living map of key AI/ML concepts - grounded in Microsoft AI terminology.
    </p>
    <p style="margin-top: 0.5rem; font-size: 0.95rem; color: #666;">
      By Daniel McLoughlin ☁️ &nbsp;|&nbsp;
      <a href="https://daniel.mcloughlin.cloud/" target="_blank" style="color: #4da6ff; text-decoration: none;">
        Visit my blog →
      </a>
      |&nbsp;
      <a href="https://github.com/CloudDevDan/dans-ai-terminology-tracker" target="_blank" style="color: #4da6ff; text-decoration: none;">
        Project repo →
      </a>
    </p>
    <p style="margin-top: 1.5rem; font-size: 1rem; color: white;">
      Click a node 🟠 to get started.
    </p>
  </header>
</div>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    const toggleBtn = document.getElementById('toggleHeaderButton');
    const headerWrapper = document.getElementById('headerWrapper');
    toggleBtn.addEventListener('click', function() {
      if (headerWrapper.style.display === 'none') {
        headerWrapper.style.display = '';
        toggleBtn.textContent = 'Hide Header';
      } else {
        headerWrapper.style.display = 'none';
        toggleBtn.textContent = 'Show Header';
      }
    });
  });
</script>
`;

html = html.replace('<body style="background-color: #1a1b26;">', `<body style="background-color: #1a1b26;">\n${headerHTML}`);

// ✅ Save file
try {
  fs.writeFileSync(path, html);
  console.log(`✅ Post-processing complete. Updated: ${path}`);
} catch (err) {
  console.error(`❌ Error writing ${path}:`, err.message);
}
