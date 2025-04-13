const fs = require('fs');
const path = require('path');

const docsDir = './docs';
const indexPath = path.join(docsDir, 'index.html');
const sectionsDir = path.join(docsDir, 'sections');

function generateMenuHtml(currentFilePath) {
  const inSection = currentFilePath.includes('/sections/');
  const relToRoot = inSection ? '../' : '';
  const relToSections = inSection ? '' : 'sections/';

  const sectionFiles = fs.existsSync(sectionsDir)
    ? fs.readdirSync(sectionsDir).filter(f => f.endsWith('.html'))
    : [];

  let html = `
<!-- MENU START -->
<style>
  nav {
    margin-top: 1rem;
    margin-bottom: 2rem;
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: center;
  }
  nav a {
    text-decoration: none;
    font-weight: 500;
    background: #f2f4f8;
    padding: 6px 14px;
    border-radius: 8px;
    color: #0078d4;
    transition: background 0.2s;
  }
  nav a:hover {
    background: #e0e6ee;
  }
</style>
<nav>
  <a href="${relToRoot}index.html">🧠 All Terms</a>`;

  sectionFiles.forEach(file => {
    const filePath = path.join(sectionsDir, file);
    const content = fs.readFileSync(filePath, 'utf-8');
    const match = content.match(/<text[^>]*>(.*?)<\/text>/i);
    const label = match ? match[1].replace(/<\/?[^>]+(>|$)/g, '') : file.replace('.html', '');
    const labelClean = label.charAt(0).toUpperCase() + label.slice(1);
    html += `<a href="${relToSections}${file}">${labelClean}</a>`;
  });

  html += '</nav>\n<!-- MENU END -->';
  return html;
}

function postprocessHtmlFile(filepath) {
  let html = fs.readFileSync(filepath, 'utf-8');

  html = html.replace(/<!-- MENU START -->[\s\S]*?<!-- MENU END -->/, '');

  if (!html.includes('apple-touch-icon')) {
    html = html.replace('</head>', `<link rel="icon" type="image/png" href="favicon/favicon.png">
<link rel="apple-touch-icon" sizes="180x180" href="favicon/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="favicon/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="favicon/favicon-16x16.png">
<link rel="manifest" href="favicon/site.webmanifest">
</head>`);
  }

  html = html.replace(/<title>.*?<\/title>/, '<title>Dan’s AI Terminology Tracker</title>');

  html = html.replace(/#mindmap\s*\{[^}]*\}/, match => {
    let updated = match;
    if (!updated.includes('color')) updated = updated.replace('}', '  color: white;\n}');
    if (!updated.includes('background-color')) updated = updated.replace('}', '  background-color: #1a1b26;\n}');
    return updated;
  });

  html = html.replace(/body\s*\{[^}]*\}/, match => {
    return match.includes('background-color') ? match : match.replace('}', '  background-color: #1a1b26;\n}');
  });

  html = html.replace(/<body[^>]*>/, '<body style="background-color: #1a1b26;">');

  const headerHTML = `
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
  body { font-family: 'Inter', sans-serif; }
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
  html = html.replace('</header>', `${generateMenuHtml(filepath)}</header>`);

  fs.writeFileSync(filepath, html, 'utf-8');
  console.log(`✅ Processed: ${filepath}`);
}

const allHtmlFiles = [
  indexPath,
  ...(fs.existsSync(sectionsDir)
    ? fs.readdirSync(sectionsDir)
        .filter(f => f.endsWith('.html'))
        .map(f => path.join(sectionsDir, f))
    : [])
];

allHtmlFiles.forEach(postprocessHtmlFile);

console.log('\n🎯 Navigation menus updated for local + hosted use!');
