import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import axios from "axios";
import { 
  schemesData, rightsData, assistiveData, 
  accessibilityData, employmentData, educationData 
} from "../data/infoData";

const AuthContext = createContext(null);

const buildApiCandidates = () => {
  const envUrl = (process.env.REACT_APP_API_URL || "").trim();
  const host = window.location.hostname || "localhost";
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";

  const candidates = [
    envUrl,
    `${protocol}//${host}:3000`,
    "http://localhost:3000",
  ].filter(Boolean);

  return [...new Set(candidates)];
};

const API_CANDIDATES = buildApiCandidates();
const API_BASE = API_CANDIDATES[0];

// Configure axios
  const api = axios.create({ baseURL: API_BASE });
  api.interceptors.request.use((config) => {
    const token = localStorage.getItem("disability_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });

  // Retry network-level failures once against alternative local API ports.
  api.interceptors.response.use(
    (response) => response,
    async (error) => {
      const config = error?.config;
      const isNetworkError = error?.code === "ERR_NETWORK" || !error?.response;

      if (!config || !isNetworkError || config.__retriedWithFallback) {
        return Promise.reject(error);
      }

      const currentBase = config.baseURL || api.defaults.baseURL;
      const fallbackBase = API_CANDIDATES.find((base) => base !== currentBase);

      if (!fallbackBase) {
        return Promise.reject(error);
      }

      config.__retriedWithFallback = true;
      config.baseURL = fallbackBase;
      api.defaults.baseURL = fallbackBase;
      return api.request(config);
    }
  );

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [language, setLanguage] = useState(
    localStorage.getItem("disability_language") || null
  );
  const [screenReaderEnabled, setScreenReaderEnabled] = useState(
    localStorage.getItem("screen_reader_enabled") === "true"
  );

  useEffect(() => {
    localStorage.setItem("screen_reader_enabled", screenReaderEnabled);
  }, [screenReaderEnabled]);

  useEffect(() => {
    const token = localStorage.getItem("disability_token");
    if (token) {
      api.get("/api/auth/me")
        .then(res => setUser(res.data))
        .catch(() => { localStorage.removeItem("disability_token"); })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!language && user?.preferred_language) {
      setLanguage(user.preferred_language);
      localStorage.setItem("disability_language", user.preferred_language);
    }
  }, [user, language]);

  useEffect(() => {
    const langCode =
      language === "hi" ? "hi" :
      language === "ta" ? "ta" :
      language === "te" ? "te" :
      language === "ml" ? "ml" :
      language === "kn" ? "kn" : "en";
    document.documentElement.lang = langCode;
  }, [language]);

  const toggleScreenReader = useCallback(() => {
    setScreenReaderEnabled(prev => !prev);
  }, []);

  const login = useCallback(async (email, password) => {
    const res = await api.post("/api/auth/login", { email, password });
  localStorage.setItem("disability_token", res.data.access_token);
    if (language) {
      await api.put("/api/auth/profile", { preferred_language: language });
    }
    setUser({ ...res.data.user, preferred_language: language || res.data.user.preferred_language });
    return res.data;
  }, [language]);

  const register = useCallback(async (data) => {
    const res = await api.post("/api/auth/register", {
      ...data, preferred_language: language || "en"
    });
  localStorage.setItem("disability_token", res.data.access_token);
    setUser(res.data.user);
    return res.data;
  }, [language]);

  const logout = useCallback(() => {
  localStorage.removeItem("disability_token");
    setUser(null);
  }, []);

  const selectLanguage = useCallback((lang) => {
    setLanguage(lang);
    localStorage.setItem("disability_language", lang);
    if (user) {
      api.put("/api/auth/profile", { preferred_language: lang })
        .then(res => setUser(prev => prev ? { ...prev, preferred_language: lang } : prev))
        .catch(() => {});
    }
  }, [user]);

  const resetLanguage = useCallback(() => {
    setLanguage(null);
    localStorage.removeItem("disability_language");
  }, []);

  const chatQuery = useCallback(async (question, sessionId, lang) => {
    const payload = {
      question,
      session_id: sessionId,
      language: lang || language || user?.preferred_language || "en",
    };

    if (user) {
      try {
        const authedRes = await api.post("/api/chat", payload);
        return authedRes.data;
      } catch (err) {
        const status = err?.response?.status;
        // If token/session is stale, fall back to guest chat instead of failing.
        if (status === 401 || status === 403) {
          const guestRes = await api.post("/api/chat/guest", payload);
          return guestRes.data;
        }
        throw err;
      }
    }

    const guestRes = await api.post("/api/chat/guest", payload);
    return guestRes.data;
  }, [user, language]);

  const getChatHistory = useCallback(async (sessionId) => {
    if (!sessionId || !user) {
      return [];
    }

    const res = await api.get(`/api/chat/history/${encodeURIComponent(sessionId)}`);
    return Array.isArray(res.data) ? res.data : [];
  }, [user]);

  const getSchemes = useCallback(() => Promise.resolve({ schemes: schemesData }), []);
  const getRights = useCallback(() => Promise.resolve({ rights: rightsData }), []);
  const getAssistive = useCallback(() => Promise.resolve({ assistive: assistiveData }), []);
  const getAccessibility = useCallback(() => Promise.resolve({ accessibility: accessibilityData }), []);
  const getEmployment = useCallback(() => Promise.resolve({ employment: employmentData }), []);
  const getEducation = useCallback(() => Promise.resolve({ education: educationData }), []);

  return (
    <AuthContext.Provider value={{
      user, loading, language, screenReaderEnabled, toggleScreenReader, login, register, logout,
      selectLanguage, resetLanguage, chatQuery, getSchemes, getRights,
      getAssistive, getAccessibility, getEmployment, getEducation, getChatHistory, api
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};

export { api };
