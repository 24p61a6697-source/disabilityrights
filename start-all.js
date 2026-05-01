const { spawn } = require('child_process');
const net = require('net');
const path = require('path');

function findFreePort(start = 3001, max = 3010) {
  return new Promise((resolve, reject) => {
    const tryPort = (port) => {
      if (port > max) {
        return reject(new Error(`No free port found between ${start} and ${max}`));
      }
      const server = net.createServer();
      server.unref();
      server.on('error', () => tryPort(port + 1));
      server.listen(port, '127.0.0.1', () => {
        server.close(() => resolve(port));
      });
    };
    tryPort(start);
  });
}

async function main() {
  const port = await findFreePort();
  const projectRoot = path.resolve(__dirname);
  const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
  const backendDir = path.join(projectRoot, 'backend');
  const env = { ...process.env, PORT: String(port), REACT_APP_API_URL: `http://127.0.0.1:${port}` };

  console.log(`Starting backend on http://127.0.0.1:${port}`);
  const backend = spawn(pythonExe, [
    '-m',
    'uvicorn',
    'app.main:app',
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
    '--reload',
    '--reload-dir',
    'app',
  ], {
    cwd: backendDir,
    env,
    stdio: 'inherit',
    shell: false,
  });

  console.log(`Starting frontend using REACT_APP_API_URL=http://127.0.0.1:${port}`);
  const frontend = spawn('npm', ['--prefix', './frontend', 'start'], {
    cwd: projectRoot,
    env,
    stdio: 'inherit',
    shell: true,
  });

  const shutdown = (code) => {
    if (!backend.killed) backend.kill();
    if (!frontend.killed) frontend.kill();
    process.exit(code);
  };

  backend.on('exit', (code) => shutdown(code));
  frontend.on('exit', (code) => shutdown(code));
  backend.on('error', (err) => {
    console.error('Backend failed to start:', err);
    shutdown(1);
  });
  frontend.on('error', (err) => {
    console.error('Frontend failed to start:', err);
    shutdown(1);
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
