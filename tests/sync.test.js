// Checks the sync merge logic in index.html. Run: node tests/sync.test.js
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const code = html.slice(html.indexOf('function mergeTasks'), html.indexOf('async function syncNow'));
const { mergeTasks, sharedCopy } = new Function(`${code}; return { mergeTasks, sharedCopy };`)();

const task = (id, updatedAt, extra = {}) => ({ id, title: `t${id}`, time: '09:00', updatedAt, ...extra });

// The newer edit wins, whichever device made it
let merged = mergeTasks([task(1, 100, { completed: true })], [task(1, 50, { completed: false })]);
assert.strictEqual(merged[0].completed, true);
merged = mergeTasks([task(1, 50, { completed: false })], [task(1, 100, { completed: true })]);
assert.strictEqual(merged[0].completed, true);

// Tasks from either side are kept; reminder history stays on this device
merged = mergeTasks([task(1, 10, { lastNotified: 'x' })], [task(1, 20), task(2, 5)]);
assert.deepStrictEqual(merged.map(t => t.id).sort(), [1, 2]);
assert.strictEqual(merged.find(t => t.id === 1).lastNotified, 'x');
assert.strictEqual(merged.find(t => t.id === 2).lastNotified, null);

// A deletion on another device removes the task here
merged = mergeTasks([task(1, 10)], [task(1, 20, { deleted: true })]);
assert.strictEqual(merged[0].deleted, true);

// Uploaded copy drops reminder history and deletions older than 30 days
const old = Date.now() - 31 * 24 * 60 * 60 * 1000;
const shared = sharedCopy([task(2, Date.now(), { lastNotified: 'x' }), task(1, old, { deleted: true }), task(3, Date.now(), { deleted: true })]);
assert.deepStrictEqual(shared.map(t => t.id), [2, 3]);
assert.ok(!('lastNotified' in shared[0]));

console.log('sync tests passed');
