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
      server.on('error', () => {
        tryPort(port + 1);
      });
      server.listen(port, '127.0.0.1', () => {
        server.close(() => resolve(port));
      });
    };
    tryPort(start);
  });
}

async function main() {
  const reload = process.argv.includes('--reload');
  const port = await findFreePort();
  const projectRoot = path.resolve(__dirname);
  const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
  const backendDir = path.join(projectRoot, 'backend');

  const args = [
    '-m',
    'uvicorn',
    'app.main:app',
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
  ];

  if (reload) {
    args.push('--reload', '--reload-dir', 'app');
  }

  const env = { ...process.env, PORT: String(port) };

  console.log(`Starting backend on http://127.0.0.1:${port}`);
  const child = spawn(pythonExe, args, {
    cwd: backendDir,
    env,
    stdio: 'inherit',
    shell: false,
  });

  child.on('exit', (code) => {
    process.exit(code);
  });

  child.on('error', (err) => {
    console.error('Failed to start backend:', err);
    process.exit(1);
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
