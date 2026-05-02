import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles/variables.css";
import "./index.css";

/* ---------------- SAFE ROOT ---------------- */

const container = document.getElementById("root");

if (!container) {
  throw new Error("Root element not found");
}

const root = ReactDOM.createRoot(container);

/* ---------------- INITIAL ACCESSIBILITY ---------------- */

// Set safe defaults (will be overridden by app)
document.documentElement.lang = navigator.language || "en";
document.title = "Disability Rights";

/* ---------------- RENDER ---------------- */

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);