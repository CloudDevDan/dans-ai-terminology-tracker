const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const mainMarkdown = 'markmap.md';
const outputMainHtml = './docs/index.html';

const inputSectionsDir = './sections'; // your .md files
const outputSectionsDir = './docs/sections';  // where .html files go

// Step 1: Clean output directory
console.log('🧹 Cleaning old HTML files in ./docs/sections...');
if (fs.existsSync(outputSectionsDir)) {
  fs.readdirSync(outputSectionsDir).forEach(file => {
    if (file.endsWith('.html')) {
      fs.unlinkSync(path.join(outputSectionsDir, file));
      console.log(`❌ Deleted: ${file}`);
    }
  });
} else {
  console.log('📁 Creating output directory ./docs/sections...');
  fs.mkdirSync(outputSectionsDir, { recursive: true });
}

// Step 2: Render the full map
console.log('\\n📘 Rendering full map...');
execSync(`npx markmap-cli "${mainMarkdown}" -o "${outputMainHtml}" --no-open`);
console.log(`✅ Full map rendered to ${outputMainHtml}`);

// Step 3: Render each section file from input dir
console.log('\\n📂 Rendering section maps...');
if (!fs.existsSync(inputSectionsDir)) {
  console.error('❌ Input directory not found:', inputSectionsDir);
  process.exit(1);
}

fs.readdirSync(inputSectionsDir).forEach(file => {
  if (!file.endsWith('.md')) return;

  const inputPath = path.join(inputSectionsDir, file);
  const outputPath = path.join(outputSectionsDir, file.replace('.md', '.html'));

  console.log(`  → ${file} → ${path.basename(outputPath)}`);
  execSync(`npx markmap-cli "${inputPath}" -o "${outputPath}" --no-open`);
});

console.log('\\n✅ All maps rendered!');
