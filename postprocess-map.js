const fs = require('fs');
const path = './docs/index.html'; // Path to your Markmap export

let html;
try {
  html = fs.readFileSync(path, 'utf8');
} catch (err) {
  console.error(`❌ Error reading ${path}:`, err.message);
  process.exit(1);
}

// ✅ Inject favicon if missing
const faviconTag = `<link rel="icon" type="image/png" href="favicon.png">`;
if (!html.includes('favicon.png')) {
  html = html.replace('</head>', `${faviconTag}\n</head>`);
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

// ✅ Inject styled header block
const headerHTML = `
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
  body {
    font-family: 'Inter', sans-serif;
  }
</style>
<header style="
  padding: 2rem 1rem;
  background-color: #0d0d0d;
  color: white;
  text-align: center;
  border-bottom: 1px solid #222;
">
  <h1 style="font-size: 2.25rem; margin: 0; font-weight: 600;">
    Dan’s AI Terminology Tracker
  </h1>
  <p style="margin-top: 0.75rem; font-size: 1.05rem; color: #aaa; max-width: 800px; margin-left: auto; margin-right: auto;">
    A living map of key AI/ML concepts - grounded in Microsoft AI terminology.
  </p>
  <p style="margin-top: 0.5rem; font-size: 0.95rem; color: #666;">
    Daniel McLoughlin ☁️ &nbsp;|&nbsp;
    <a href="https://daniel.mcloughlin.cloud/" target="_blank" style="color: #4da6ff; text-decoration: none;">
      Visit my blog →
    </a>
  </p>
</header>
`;

html = html.replace('<body style="background-color: #1a1b26;">', `<body style="background-color: #1a1b26;">\n${headerHTML}`);

// ✅ Save file
try {
  fs.writeFileSync(path, html);
  console.log(`✅ Post-processing complete. Updated: ${path}`);
} catch (err) {
  console.error(`❌ Error writing ${path}:`, err.message);
}
