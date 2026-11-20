// Writes styles/<look>/index.html: the app locked to one look.
// Run after changing index.html: node scripts/build-styles.js
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const LOOKS = {
  skeuomorphism: ['skeuo', 'Skeuomorphism'],
  neumorphism: ['neu', 'Neumorphism'],
  glassmorphism: ['glass', 'Glassmorphism'],
  claymorphism: ['clay', 'Claymorphism'],
  brutalism: ['brutal', 'Brutalism'],
  maximalism: ['max', 'Maximalism'],
  'liquid-glass': ['liquid', 'Liquid Glass'],
  'bento-grid': ['bento', 'Bento Grid'],
  'spatial-ui': ['spatial', 'Spatial UI'],
};

const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const loader = html.match(/  <script>\n    try \{ const st = localStorage[^\n]*\n  <\/script>\n/);
if (!loader) throw new Error('Saved-look loader not found in index.html');

for (const [folder, [id, name]] of Object.entries(LOOKS)) {
  const page = html
    .replace('<html lang="en">', `<html lang="en" data-style="${id}">`)
    .replace('<title>Daily Task Automator</title>', `<title>Daily Task Automator · ${name}</title>`)
    .replace(loader[0], '  <style>#lookPanel { display: none; }</style>\n')
    .replace('href="manifest.webmanifest"', 'href="../../manifest.webmanifest"')
    .replace('href="icon-192.png"', 'href="../../icon-192.png"');
  const dir = path.join(root, 'styles', folder);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'index.html'), page);
}
console.log(`Wrote ${Object.keys(LOOKS).length} looks to styles/`);
