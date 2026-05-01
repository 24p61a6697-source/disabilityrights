// Simple Express server to serve the React build folder
const express = require('express');
const path = require('path');

const app = express();
const PORT = 3000;

// Serve static files from the build directory
app.use(express.static(path.join(__dirname, 'frontend', 'build')));

// Catch-all route to serve index.html for client-side routing
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'build', 'index.html'));
});

app.listen(PORT, '127.0.0.1', () => {
  console.log(`Frontend server running at http://127.0.0.1:${PORT}`);
  console.log(`Backend API at http://127.0.0.1:9000`);
});
