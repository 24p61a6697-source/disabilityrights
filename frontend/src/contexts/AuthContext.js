import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  schemesData, rightsData, assistiveData,
  accessibilityData, employmentData, educationData
} from "../data/infoData";

const AuthContext = createContext(null);

const isBrowser = typeof window !== "undefined";

/* ---------------- API CONFIG ---------------- */

function buildApiCandidates() {
  if (!isBrowser) return ["http://localhost:3000"];

  const envUrl = (process.env.REACT_APP_API_URL || "").trim();
  const host = window.location.hostname || "localhost";
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";

  return [...new Set([
    envUrl,
    `${protocol}//${host}:8000`,
    "http://localhost:8000",
    `${protocol}//${host}:3000`,
    "http://localhost:3000",
  ].filter(Boolean))];
}

const API_CANDIDATES = buildApiCandidates();

const api = axios.create({
  baseURL: API_CANDIDATES[0],
  timeout: 8000, // ✅ important
});

/* ---------------- INTERCEPTORS ---------------- */

api.interceptors.request.use((config) => {
  if (isBrowser) {
    const token = localStorage.getItem("disability_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const config = error?.config;

    if (!config || config.__retry) {
      return Promise.reject(error);
    }

    const isNetworkError = !error.response;

    if (isNetworkError) {
      const fallback = API_CANDIDATES.find(
        (url) => url !== config.baseURL
      );

      if (fallback) {
        config.__retry = true;
        config.baseURL = fallback;
        api.defaults.baseURL = fallback;
        return api.request(config);
      }
    }

   
    if (error?.response?.status === 401 && isBrowser) {
      localStorage.removeItem("disability_token");
    }

    return Promise.reject(error);
  }
);

/* ---------------- PROVIDER ---------------- */

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [language, setLanguage] = useState(
    isBrowser ? localStorage.getItem("disability_language") : null
  );
  const [screenReaderEnabled, setScreenReaderEnabled] = useState(
    isBrowser && localStorage.getItem("screen_reader_enabled") === "true"
  );

  /* ---------- PERSIST ---------- */

  useEffect(() => {
    if (isBrowser) {
      localStorage.setItem("screen_reader_enabled", screenReaderEnabled);
    }
  }, [screenReaderEnabled]);

  /* ---------- AUTH INIT ---------- */

  useEffect(() => {
    if (!isBrowser) return;

    const token = localStorage.getItem("disability_token");

    if (!token) {
      setLoading(false);
      return;
    }

    api.get("/api/auth/me")
      .then(res => setUser(res.data))
      .catch(() => {
        localStorage.removeItem("disability_token");
        setUser(null);
      })
      .finally(() => setLoading(false));
  }, []);

  /* ---------- LANGUAGE ---------- */

  useEffect(() => {
    if (!language && user?.preferred_language) {
      setLanguage(user.preferred_language);
      localStorage.setItem("disability_language", user.preferred_language);
    }
  }, [user, language]);

  useEffect(() => {
    const lang = language || "en";
    document.documentElement.lang = lang;
  }, [language]);

  /* ---------- ACTIONS ---------- */

  const toggleScreenReader = useCallback(() => {
    setScreenReaderEnabled(v => !v);
  }, []);

  const login = useCallback(async (email, password) => {
    const res = await api.post("/api/auth/login", { email, password });

    if (isBrowser) {
      localStorage.setItem("disability_token", res.data.access_token);
    }

    setUser(res.data.user);
    return res.data;
  }, []);

  const register = useCallback(async (data) => {
    const res = await api.post("/api/auth/register", {
      ...data,
      preferred_language: language || "en",
    });

    if (isBrowser) {
      localStorage.setItem("disability_token", res.data.access_token);
    }

    setUser(res.data.user);
    return res.data;
  }, [language]);

  const logout = useCallback(() => {
    if (isBrowser) {
      localStorage.removeItem("disability_token");
    }
    setUser(null);
  }, []);

  const selectLanguage = useCallback((lang) => {
    setLanguage(lang);

    if (isBrowser) {
      localStorage.setItem("disability_language", lang);
    }

    if (user) {
      api.put("/api/auth/profile", { preferred_language: lang })
        .then(() => {
          setUser(prev => prev ? { ...prev, preferred_language: lang } : prev);
        })
        .catch((err) => {
          console.error("Language update failed:", err);
        });
    }
  }, [user]);

  const resetLanguage = useCallback(() => {
    setLanguage(null);
    if (isBrowser) {
      localStorage.removeItem("disability_language");
    }
  }, []);

  /* ---------- CHAT ---------- */

  const chatQuery = useCallback(async (question, sessionId, lang) => {
    const payload = {
      question,
      session_id: sessionId,
      language: lang || language || user?.preferred_language || "en",
    };

    try {
      if (user) {
        const res = await api.post("/api/chat", payload);
        return res.data;
      }
    } catch (err) {
      if (err?.response?.status === 401) {
        const guestRes = await api.post("/api/chat/guest", payload);
        return guestRes.data;
      }
      throw err;
    }

    const guestRes = await api.post("/api/chat/guest", payload);
    return guestRes.data;
  }, [user, language]);

  const getChatHistory = useCallback(async (sessionId) => {
    if (!sessionId || !user) return [];

    const res = await api.get(`/api/chat/history/${encodeURIComponent(sessionId)}`);
    return Array.isArray(res.data) ? res.data : [];
  }, [user]);

  /* ---------- STATIC DATA ---------- */

  const getSchemes = () => Promise.resolve({ schemes: schemesData });
  const getRights = () => Promise.resolve({ rights: rightsData });
  const getAssistive = () => Promise.resolve({ assistive: assistiveData });
  const getAccessibility = () => Promise.resolve({ accessibility: accessibilityData });
  const getEmployment = () => Promise.resolve({ employment: employmentData });
  const getEducation = () => Promise.resolve({ education: educationData });

  return (
    <AuthContext.Provider value={{
      user, loading, language, screenReaderEnabled,
      toggleScreenReader, login, register, logout,
      selectLanguage, resetLanguage, chatQuery,
      getSchemes, getRights, getAssistive,
      getAccessibility, getEmployment, getEducation,
      getChatHistory, api
    }}>
      {children}
    </AuthContext.Provider>
  );
}

/* ---------------- HOOK ---------------- */

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};

export { api };