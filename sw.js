// Lets phones show reminder notifications; tapping one opens the app.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));

self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(windows =>
      windows.length ? windows[0].focus() : self.clients.openWindow(self.registration.scope))
  );
});
