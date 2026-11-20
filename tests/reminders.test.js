// Checks when reminders fire. Run: node tests/reminders.test.js
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const code = html.slice(html.indexOf('// ---- Reminder rules'), html.indexOf('// ---- end reminder rules'));
const { isDue, isDone } = new Function(`${code}; return { isDue, isDone };`)();

const fri = (h, m) => new Date(2026, 10, 20, h, m); // Friday 20 Nov 2026
const created = new Date(2026, 10, 1).getTime();
const task = extra => ({ id: created, title: 'x', time: '20:00', repeat: 'daily', ...extra });

// Fires at its minute, and up to an hour late if the phone paused the page
assert.ok(!isDue(task(), fri(19, 59)));
assert.ok(isDue(task(), fri(20, 0)));
assert.ok(isDue(task(), fri(20, 40)));
assert.ok(!isDue(task(), fri(21, 0)));

// Only once a day; a new day fires again
assert.ok(!isDue(task({ lastNotified: fri(20, 0).toISOString() }), fri(20, 5)));
assert.ok(isDue(task({ lastNotified: new Date(2026, 10, 19, 20, 0).toISOString() }), fri(20, 5)));
assert.ok(!isDue(task({ repeat: 'once', lastNotified: new Date(2026, 10, 19, 20, 0).toISOString() }), fri(20, 5)));

// Repeat days: Friday is a weekday
assert.ok(isDue(task({ repeat: 'weekdays' }), fri(20, 0)));
assert.ok(!isDue(task({ repeat: 'weekends' }), fri(20, 0)));

// A daily task ticked off yesterday is due again today; ticked today is not
assert.ok(isDue(task({ completed: true, completedOn: '2026-11-19' }), fri(20, 0)));
assert.ok(!isDue(task({ completed: true, completedOn: '2026-11-20' }), fri(20, 0)));
assert.ok(!isDone(task({ completed: true, completedOn: '2026-11-19' }), fri(9, 0)));
assert.ok(isDone(task({ repeat: 'once', completed: true }), fri(9, 0)));

// A task added after its time today waits for tomorrow
assert.ok(!isDue(task({ id: fri(20, 10).getTime() }), fri(20, 15)));

console.log('reminder tests passed');
