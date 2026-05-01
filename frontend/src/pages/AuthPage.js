import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { t, INDIAN_LANGUAGES } from "../languages/languages";
import { motion, AnimatePresence } from "framer-motion";
import { 
  AlertCircle, 
  ArrowLeft
} from "lucide-react";
import useDomScreenReader from "../hooks/useDomScreenReader";

const STATES = [
  "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
  "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka",
  "Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram",
  "Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana",
  "Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
  "Andaman & Nicobar Islands","Chandigarh","Dadra & Nagar Haveli","Daman & Diu",
  "Delhi","Jammu & Kashmir","Ladakh","Lakshadweep","Puducherry"
];

const DISABILITY_TYPES = [
  "Blindness","Low Vision","Deaf (Hearing Impairment)","Hard of Hearing",
  "Speech & Language Disability","Locomotor Disability","Cerebral Palsy",
  "Dwarfism","Intellectual Disability","Autism Spectrum Disorder",
  "Specific Learning Disabilities (Dyslexia/Dyscalculia/Dysgraphia)",
  "Mental Illness","Muscular Dystrophy","Multiple Sclerosis",
  "Haemophilia","Thalassemia","Sickle Cell Disease","Acid Attack Survivor",
  "Multiple Disabilities","Other / Not Specified","Prefer not to say"
];

export default function AuthPage() {
  const { login, register, language, selectLanguage } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState("login"); // login | register
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [screenReaderOn, setScreenReaderOn] = useState(false);
  const tr = (k) => t(k, language);

  useDomScreenReader(screenReaderOn, language || "en");

  const [loginForm, setLoginForm] = useState({ email: "", password: "" });
  const [regForm, setRegForm] = useState({
    full_name: "", email: "", mobile: "", state: "", disability_type: "", password: "", confirm_password: ""
  });

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(loginForm.email, loginForm.password);
      navigate("/chat");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to sign in");
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError("");
    if (regForm.password !== regForm.confirm_password) {
      setError("Passwords do not match");
      return;
    }
    setLoading(true);
    try {
      await register({
        full_name: regForm.full_name.trim(),
        email: regForm.email.trim(),
        mobile: regForm.mobile.trim(),
        state: regForm.state,
        disability_type: regForm.disability_type,
        password: regForm.password,
      });
      navigate("/chat");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to create account");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <header className="gov-header">
        <div className="container gov-header-stack">
          <div className="brand" onClick={() => navigate("/")}>
            <span className="emblem">♿</span>
            <div className="brand-text">
              <div className="govt-title">Disability Rights</div>
              <div className="govt-subtitle">{tr("appSubtitle")}</div>
            </div>
          </div>

          <div className="header-tools header-tools-auth">
            <button onClick={() => navigate("/")} className="header-pill">
              <ArrowLeft size={16} />
              <span>{tr("home")}</span>
            </button>
            <button
              className="header-pill"
              onClick={() => setScreenReaderOn((v) => !v)}
              aria-pressed={screenReaderOn}
            >
              {screenReaderOn ? `🔊 ${tr("screenReaderOn")}` : `🔇 ${tr("screenReaderOff")}`}
            </button>
            <label className="header-lang-select" aria-label="Language selector">
              <span>{tr("changeLanguage")}:</span>
              <select value={language || "en"} onChange={(e) => selectLanguage(e.target.value)}>
                {INDIAN_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>{lang.english} ({lang.name})</option>
                ))}
              </select>
            </label>
          </div>
        </div>
        <div className="tricolor-bar">
          <div className="saffron" />
          <div className="white" />
          <div className="green" />
        </div>
      </header>

      <main className="container auth-main">
        <div className="auth-split auth-centered">
          <div className="auth-page-title">♿ Disability Rights</div>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="auth-card premium-card"
          >
            <div className="auth-tabs">
              <button 
                className={mode === "login" ? "active" : ""} 
                onClick={() => setMode("login")}
              >
                {tr("login")}
              </button>
              <button 
                className={mode === "register" ? "active" : ""} 
                onClick={() => setMode("register")}
              >
                {tr("register")}
              </button>
            </div>

            <div className="auth-body">
              <AnimatePresence mode="wait">
                {error && (
                  <motion.div 
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="error-banner"
                  >
                    <AlertCircle size={16} />
                    <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              {mode === "login" ? (
                <motion.form 
                  key="login"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  onSubmit={handleLogin}
                  className="auth-form"
                >
                  <h2 className="auth-form-title">{tr("welcomeBack")}</h2>
                  <p className="auth-form-subtitle">Sign in to access your disability rights dashboard</p>
                  <div className="input-field">
                    <label>{tr("email")} *</label>
                    <input 
                      type="email" 
                      placeholder={tr("email")}
                      value={loginForm.email}
                      onChange={e => setLoginForm(f => ({...f, email: e.target.value}))}
                      required 
                    />
                  </div>
                  <div className="input-field">
                    <label>{tr("password")} *</label>
                    <input 
                      type="password" 
                      placeholder={tr("password")}
                      value={loginForm.password}
                      onChange={e => setLoginForm(f => ({...f, password: e.target.value}))}
                      required 
                    />
                  </div>
                  <button type="submit" className="submit-btn" disabled={loading}>
                    {loading ? "Signing in..." : tr("loginBtn")}
                  </button>
                  <p className="auth-switch-helper">{tr("noAccount")} <button type="button" className="link-like" onClick={() => setMode("register")}>{tr("register")}</button></p>
                </motion.form>
              ) : (
                <motion.form 
                  key="register"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  onSubmit={handleRegister}
                  className="auth-form"
                >
                  <h2 className="auth-form-title">{tr("createAccount")}</h2>
                  <div className="form-grid">
                    <div className="input-field">
                      <label>{tr("fullName")}</label>
                      <input 
                        type="text" 
                        placeholder={tr("fullName")}
                        value={regForm.full_name}
                        onChange={e => setRegForm(f => ({...f, full_name: e.target.value}))}
                        required 
                      />
                    </div>
                    <div className="input-field">
                      <label>{tr("email")}</label>
                      <input 
                        type="email" 
                        placeholder={tr("email")}
                        value={regForm.email}
                        onChange={e => setRegForm(f => ({...f, email: e.target.value}))}
                        required 
                      />
                    </div>
                    <div className="input-field">
                      <label>{tr("mobile")}</label>
                      <input 
                        type="tel" 
                        placeholder={tr("mobile")}
                        maxLength={10}
                        value={regForm.mobile}
                        onChange={e => setRegForm(f => ({...f, mobile: e.target.value.replace(/\D/g,"")}))}
                        required 
                      />
                    </div>
                    <div className="input-field">
                      <label>{tr("state")}</label>
                      <select 
                        value={regForm.state}
                        onChange={e => setRegForm(f => ({...f, state: e.target.value}))}
                      >
                        <option value="">Select State</option>
                        {STATES.map(s => <option key={s} value={s}>{s}</option>)}
                      </select>
                    </div>
                    <div className="input-field full">
                      <label>{tr("disabilityType")}</label>
                      <select 
                        value={regForm.disability_type}
                        onChange={e => setRegForm(f => ({...f, disability_type: e.target.value}))}
                      >
                        <option value="">{tr("disabilityType")} (Optional)</option>
                        {DISABILITY_TYPES.map(d => <option key={d} value={d}>{d}</option>)}
                      </select>
                    </div>
                    <div className="input-field">
                      <label>{tr("password")}</label>
                      <input 
                        type="password" 
                        placeholder={tr("password")}
                        value={regForm.password}
                        onChange={e => setRegForm(f => ({...f, password: e.target.value}))}
                        required 
                        minLength={6}
                      />
                    </div>
                    <div className="input-field">
                      <label>{tr("confirmPassword")}</label>
                      <input 
                        type="password" 
                        placeholder={tr("confirmPassword")}
                        value={regForm.confirm_password}
                        onChange={e => setRegForm(f => ({...f, confirm_password: e.target.value}))}
                        required 
                      />
                    </div>
                  </div>
                  <button type="submit" className="submit-btn" disabled={loading}>
                    {loading ? "Creating Account..." : tr("registerBtn")}
                  </button>
                  <p className="auth-switch-helper">{tr("haveAccount")} <button type="button" className="link-like" onClick={() => setMode("login")}>{tr("login")}</button></p>
                </motion.form>
              )}

            </div>
          </motion.div>
          <div className="a11y-compact-note">♿ Keyboard navigable · Screen reader compatible (NVDA, JAWS) · Supports NVDA (NonVisual Desktop Access) · WCAG 2.0 AA compliant</div>
        </div>
      </main>
    </div>
  );
}

 