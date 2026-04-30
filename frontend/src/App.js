import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import LanguageSelect from "./pages/LanguageSelect";
import AuthPage from "./pages/AuthPage";
import ChatPage from "./pages/ChatPage";
import useDomScreenReader from "./hooks/useDomScreenReader";

function ProtectedChatRoute() {
  // Allow access to chat - even guests can chat
  return <ChatPage />;
}

function AppRoutes() {
  const { screenReaderEnabled, language, user } = useAuth();
  const currentLang = language || user?.preferred_language || "en";
  useDomScreenReader(screenReaderEnabled, currentLang);

  return (
    <Routes>
      <Route path="/" element={<LanguageSelect />} />
      <Route path="/auth" element={<AuthPage />} />
      <Route path="/chat" element={<ProtectedChatRoute />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  );
}
