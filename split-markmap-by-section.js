const fs = require('fs');
const path = require('path');

const inputPath = 'markmap.md'; // Your main file
const outputDir = './sections'; // Where the section files will go

// Read the full file
const content = fs.readFileSync(inputPath, 'utf-8');

// Match all sections by ## headings
const sectionRegex = /^## (.+)$/gm;
let match;
let sections = [];

// Find all section headings
while ((match = sectionRegex.exec(content)) !== null) {
  sections.push({ title: match[1], index: match.index });
}

// Add end index for last section
sections.forEach((sec, i) => {
  sec.end = (i < sections.length - 1) ? sections[i + 1].index : content.length;
  sec.body = content.slice(sec.index, sec.end).trim();
});

// Ensure output directory exists
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);

// Write each section to a separate .md file
sections.forEach(sec => {
  const filename = sec.title.toLowerCase().replace(/[^\w]+/g, '-').replace(/^-|-$/g, '') + '.md';
  const fullPath = path.join(outputDir, filename);
  fs.writeFileSync(fullPath, sec.body, 'utf-8');
  console.log(`✅ Created: ${fullPath}`);
});

console.log(`\n📁 Done! ${sections.length} section files saved in: ${outputDir}`);