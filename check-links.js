// check-links.js
const fs = require('fs');
const https = require('https');
const readline = require('readline');

const markdownFile = 'markmap.md';
const logFile = 'broken-links.log';

const trusted403 = new Set([
  'https://research.ibm.com/blog/what-is-federated-learning',
  'https://machinelearningmastery.com/the-attention-mechanism-from-scratch/'
]);

let currentSection = '';
const links = [];
const brokenLinks = [];

const rl = readline.createInterface({
  input: fs.createReadStream(markdownFile),
  crlfDelay: Infinity
});

rl.on('line', (line) => {
  const h2 = line.match(/^## (.+)/);
  const h3 = line.match(/^### (.+)/);
  if (h2) currentSection = h2[1];
  if (h3) currentSection += ' / ' + h3[1];

  const linkMatch = line.match(/<a\s+href="(https:\/\/[^"]+)"[^>]*>([^<]+)<\/a>/);
  if (linkMatch) {
    const url = linkMatch[1];
    const term = linkMatch[2];
    links.push({ url, term, section: currentSection });
  }
});

rl.on('close', async () => {
  console.log(`🔍 Checking ${links.length} links...\n`);
  for (const { url, term, section } of links) {
    await checkUrl(url, term, section);
  }

  if (brokenLinks.length > 0) {
    fs.writeFileSync(
      logFile,
      brokenLinks
        .map(entry => `${entry.section}:\n  ${entry.status} ${entry.term} → ${entry.url}\n`)
        .join('\n')
    );
    console.log(`\n📄 Broken links written to ${logFile}`);
  } else {
    if (fs.existsSync(logFile)) {
      fs.writeFileSync(logFile, ''); // Clear out old log
      console.log(`✅ No broken links found. Cleared existing ${logFile}`);
    } else {
      console.log(`✅ No broken links found.`);
    }
  }  
});

function checkUrl(url, term, section) {
  return new Promise(resolve => {
    const options = new URL(url);
    options.headers = {
      'User-Agent': 'Mozilla/5.0 (LinkChecker)'
    };

    https.get(options, res => {
      const { statusCode } = res;
      const isBroken = (statusCode >= 400 && !trusted403.has(url));
      const statusText = isBroken ? '❌' : '✅';
      console.log(`${section}:\n  ${statusText} [${statusCode}] ${term} → ${url}${trusted403.has(url) ? ' (trusted override)' : ''}\n`);
      if (isBroken) {
        brokenLinks.push({ url, term, section, status: `[${statusCode}]` });
      }
      res.resume();
      resolve();
    }).on('error', err => {
      console.log(`${section}:\n  ❌ [ERROR] ${term} → ${url} (${err.message})\n`);
      brokenLinks.push({ url, term, section, status: '[ERROR]' });
      resolve();
    });
  });
}
