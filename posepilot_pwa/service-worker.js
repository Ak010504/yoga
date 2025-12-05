self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", () => clients.claim());

self.addEventListener("fetch", (event) => {
  // Always fetch fresh files — do NOT use cache
  event.respondWith(fetch(event.request));
});
