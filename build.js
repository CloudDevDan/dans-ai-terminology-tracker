const { execSync } = require('child_process');
const path = require('path');

function runStep(name, command) {
  console.log(`\n▶️ ${name}`);
  try {
    execSync(command, { stdio: 'inherit' });
    console.log(`✅ ${name} completed`);
  } catch (err) {
    console.error(`❌ ${name} failed:\n`, err.message);
    process.exit(1);
  }
}

// Step 1: Split markmap.md into section .md files
runStep('Splitting markmap.md into sections', 'node split-markmap-by-section.js');

// Step 2: Render all .md files into HTML
runStep('Rendering full + section maps to HTML', 'node render-all.js');

// Step 3: Postprocess all HTML files with header and menu
runStep('Postprocessing HTML with header + menu', 'node postprocess-map.js');

console.log('\n🏁 Build finished successfully!');