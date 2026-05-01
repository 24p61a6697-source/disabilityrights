import React, { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { t, INDIAN_LANGUAGES } from "../languages/languages";
import { motion, AnimatePresence } from "framer-motion";
import { 
  MessageSquare, BookOpen, Scale, Accessibility, 
  Briefcase, GraduationCap, Plus, 
  Send, Globe, Award, ExternalLink, History,
  User as UserIcon, Keyboard, Volume2, Mic, MicOff, Square
} from "lucide-react";
import { getSpeechLocale, getVoiceAvailability, speakTextInLanguage, stopSpeech } from "../utils/speechUtils";

const CATEGORY_STYLES = {
  rpwd_act: { bg: "#e0f2fe", color: "#0369a1", labelKey: "badgeRpwdAct", icon: "⚖️" },
  employment: { bg: "#fef3c7", color: "#92400e", labelKey: "badgeEmployment", icon: "💼" },
  education: { bg: "#dcfce7", color: "#166534", labelKey: "badgeEducation", icon: "🎓" },
  schemes: { bg: "#f3e8ff", color: "#6b21a8", labelKey: "badgeScheme", icon: "🏦" },
  assistive_tech: { bg: "#ffedd5", color: "#9a3412", labelKey: "badgeAssistive", icon: "🦽" },
  accessibility: { bg: "#fce7f3", color: "#9d174d", labelKey: "badgeAccess", icon: "♿" },
  default: { bg: "#f1f5f9", color: "#475569", labelKey: "badgeNotice", icon: "ℹ️" },
};

const QUICK_TOPICS = {
  en: [
    { label: "RPWD Act Rights", query: "Explain my rights under RPWD Act 2016" },
    { label: "Job Reservation", query: "What is job reservation for persons with disabilities?" },
    { label: "Education Rights", query: "Tell me education rights for persons with disabilities" },
    { label: "Govt Schemes", query: "List key government schemes for disability support" },
    { label: "Assistive Technology", query: "Suggest assistive technology options for disability support" },
    { label: "UDID Card", query: "How do I apply for UDID card?" },
    { label: "Accessibility", query: "What accessibility standards apply in India?" },
    { label: "Disability Pension", query: "How to apply for disability pension?" },
    { label: "Free Healthcare", query: "What free healthcare support is available?" },
    { label: "File Complaint", query: "How can I file a disability rights complaint?" },
  ],
  hi: [
    { label: "आरपीडब्ल्यूडी अधिनियम अधिकार", query: "Explain my rights under RPWD Act 2016" },
    { label: "नौकरी आरक्षण", query: "What is job reservation for persons with disabilities?" },
    { label: "शिक्षा अधिकार", query: "Tell me education rights for persons with disabilities" },
    { label: "सरकारी योजनाएं", query: "List key government schemes for disability support" },
    { label: "सहायक तकनीक", query: "Suggest assistive technology options for disability support" },
    { label: "यूडीआईडी कार्ड", query: "How do I apply for UDID card?" },
    { label: "सुगम्यता", query: "What accessibility standards apply in India?" },
    { label: "दिव्यांग पेंशन", query: "How to apply for disability pension?" },
    { label: "मुफ्त स्वास्थ्य सेवा", query: "What free healthcare support is available?" },
    { label: "शिकायत दर्ज करें", query: "How can I file a disability rights complaint?" },
  ],
  ta: [
    { label: "RPWD சட்ட உரிமைகள்", query: "Explain my rights under RPWD Act 2016" },
    { label: "வேலை ஒதுக்கீடு", query: "What is job reservation for persons with disabilities?" },
    { label: "கல்வி உரிமைகள்", query: "Tell me education rights for persons with disabilities" },
    { label: "அரசு திட்டங்கள்", query: "List key government schemes for disability support" },
    { label: "உதவி தொழில்நுட்பம்", query: "Suggest assistive technology options for disability support" },
    { label: "UDID அட்டை", query: "How do I apply for UDID card?" },
    { label: "அணுகல்தன்மை", query: "What accessibility standards apply in India?" },
    { label: "மாற்றுத் திறனாளி ஓய்வூதியம்", query: "How to apply for disability pension?" },
    { label: "இலவச சுகாதாரம்", query: "What free healthcare support is available?" },
    { label: "புகார் அளிக்க", query: "How can I file a disability rights complaint?" },
  ],
  te: [
    { label: "RPWD చట్ట హక్కులు", query: "Explain my rights under RPWD Act 2016" },
    { label: "ఉద్యోగ రిజర్వేషన్", query: "What is job reservation for persons with disabilities?" },
    { label: "విద్యా హక్కులు", query: "Tell me education rights for persons with disabilities" },
    { label: "ప్రభుత్వ పథకాలు", query: "List key government schemes for disability support" },
    { label: "సహాయక సాంకేతికత", query: "Suggest assistive technology options for disability support" },
    { label: "UDID కార్డ్", query: "How do I apply for UDID card?" },
    { label: "ప్రాప్యత", query: "What accessibility standards apply in India?" },
    { label: "వికలాంగ పెన్షన్", query: "How to apply for disability pension?" },
    { label: "ఉచిత ఆరోగ్య సేవ", query: "What free healthcare support is available?" },
    { label: "ఫిర్యాదు దాఖలు", query: "How can I file a disability rights complaint?" },
  ],
  kn: [
    { label: "RPWD ಕಾಯ್ದೆ ಹಕ್ಕುಗಳು", query: "Explain my rights under RPWD Act 2016" },
    { label: "ಉದ್ಯೋಗ ಮೀಸಲಾತಿ", query: "What is job reservation for persons with disabilities?" },
    { label: "ಶಿಕ್ಷಣ ಹಕ್ಕುಗಳು", query: "Tell me education rights for persons with disabilities" },
    { label: "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು", query: "List key government schemes for disability support" },
    { label: "ಸಹಾಯಕ ತಂತ್ರಜ್ಞಾನ", query: "Suggest assistive technology options for disability support" },
    { label: "UDID ಕಾರ್ಡ್", query: "How do I apply for UDID card?" },
    { label: "ಪ್ರವೇಶಾರ್ಹತೆ", query: "What accessibility standards apply in India?" },
    { label: "ಅಂಗವೈಕಲ್ಯ ಪಿಂಚಣಿ", query: "How to apply for disability pension?" },
    { label: "ಉಚಿತ ಆರೋಗ್ಯ ಸೇವೆ", query: "What free healthcare support is available?" },
    { label: "ದೂರು ಸಲ್ಲಿಸಿ", query: "How can I file a disability rights complaint?" },
  ],
  ml: [
    { label: "RPWD നിയമ അവകാശങ്ങൾ", query: "Explain my rights under RPWD Act 2016" },
    { label: "ജോലി സംവരണം", query: "What is job reservation for persons with disabilities?" },
    { label: "വിദ്യാഭ്യാസ അവകാശങ്ങൾ", query: "Tell me education rights for persons with disabilities" },
    { label: "സർക്കാർ പദ്ധതികൾ", query: "List key government schemes for disability support" },
    { label: "സഹായ സാങ്കേതികവിദ്യ", query: "Suggest assistive technology options for disability support" },
    { label: "UDID കാർഡ്", query: "How do I apply for UDID card?" },
    { label: "പ്രവേശനക്ഷമത", query: "What accessibility standards apply in India?" },
    { label: "വൈകല്യ പെൻഷൻ", query: "How to apply for disability pension?" },
    { label: "സൗജന്യ ആരോഗ്യ സേവനം", query: "What free healthcare support is available?" },
    { label: "പരാതി നൽകുക", query: "How can I file a disability rights complaint?" },
  ],
};

const OFFICIAL_PORTALS = [
  { labelKey: "portalDepwd", url: "https://disabilityaffairs.gov.in" },
  { labelKey: "portalUdid", url: "https://www.swavlambancard.gov.in" },
  { labelKey: "portalChiefCommissioner", url: "https://ccdisabilities.nic.in" },
  { labelKey: "portalNationalTrust", url: "https://thenationaltrust.gov.in" },
  { labelKey: "portalNhfdc", url: "https://nhfdc.nic.in" },
  { labelKey: "portalAlimco", url: "https://www.alimco.in" },
];

function SourceBadge({ source, tr }) {
  const cat = source.category || "default";
  const style = CATEGORY_STYLES[cat] || CATEGORY_STYLES.default;
  return (
    <motion.span 
      whileHover={{ y: -1 }}
      className="source-badge"
      style={{ background: style.bg, color: style.color }}
    >
      <span className="source-icon" aria-hidden="true">{style.icon}</span>
      {tr(style.labelKey)}: {source.chapter || source.source}
    </motion.span>
  );
}

function Message({ msg, tr, activeSpeakingId, onToggleSpeech }) {
  const isUser = msg.role === "user";
  const isGreeting = msg.id === "greeting" || msg.id === "g";
  const isSpeaking = activeSpeakingId === msg.id;
  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      className={`message-row ${isUser ? "user" : "assistant"}`}
    >
      <div className="message-avatar" aria-hidden="true">
        {isUser ? <UserIcon size={18} /> : <Accessibility size={18} />}
      </div>
      <div className="message-content-wrapper">
        <div className={`message-bubble ${isGreeting ? "greeting-message" : ""}`}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {msg.content}
          </ReactMarkdown>
        </div>
        
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="message-sources">
            <p className="sources-label">{tr("verifiedSources")}:</p>
            <div className="sources-list">
              {msg.sources.map((src, i) => <SourceBadge key={i} source={src} tr={tr} />)}
            </div>
          </div>
        )}
        
        {!isUser && (
          <div className="message-actions">
            <button
              className={`speaker-command-btn ${isSpeaking ? "speaking-active" : ""}`}
              onClick={() => onToggleSpeech(msg.id, msg.content)}
              title={isSpeaking ? "Stop (S)" : "Read (R)"}
              aria-label={isSpeaking ? "Stop reading answer" : "Read answer aloud"}
            >
              {isSpeaking ? <Square size={14} fill="currentColor" /> : <Volume2 size={14} />}
              <span>{isSpeaking ? "Stop" : "Read"}</span>
            </button>
          </div>
        )}
      </div>
    </motion.div>
  );
}

function TypingIndicator() {
  return (
    <div className="message-row assistant">
      <div className="message-avatar typing"><Accessibility size={18} /></div>
      <div className="typing-bubble">
        <span className="dot"></span>
        <span className="dot"></span>
        <span className="dot"></span>
      </div>
    </div>
  );
}

export default function ChatPage() {
  const { 
    user, logout, chatQuery, language, selectLanguage,
    screenReaderEnabled, toggleScreenReader,
    getSchemes, getRights, getAssistive, getAccessibility, getEmployment, getEducation, getChatHistory
  } = useAuth();
  const navigate = useNavigate();

  const currentLang = language || user?.preferred_language || "en";
  const tr = useCallback((k) => t(k, currentLang), [currentLang]);

  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem("chat_history");
    return saved ? JSON.parse(saved) : [];
  });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const createSessionId = useCallback(() => `s_${Date.now()}_${Math.floor(Math.random() * 10000)}`, []);
  const [sessionId, setSessionId] = useState(() => localStorage.getItem("chat_session_id") || `s_${Date.now()}_${Math.floor(Math.random() * 10000)}`);
  const [sidebarOpen] = useState(true);
  const [activeSection, setActiveSection] = useState("chat");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [backendHistory, setBackendHistory] = useState([]);
  const [schemes, setSchemes] = useState([]);
  const [rights, setRights] = useState([]);
  const [assistive, setAssistive] = useState([]);
  const [accessibility, setAccessibility] = useState([]);
  const [employment, setEmployment] = useState([]);
  const [education, setEducation] = useState([]);
  const [selectedEducation, setSelectedEducation] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [activeSpeakingId, setActiveSpeakingId] = useState(null);

  const handleToggleSpeech = useCallback((id, content) => {
    if (activeSpeakingId === id) {
      stopSpeech();
      setActiveSpeakingId(null);
    } else {
      setActiveSpeakingId(id);
      speakTextInLanguage(content, currentLang, () => {
        setActiveSpeakingId(prev => (prev === id ? null : prev));
      });
    }
  }, [activeSpeakingId, currentLang]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (activeSection !== "chat") return;

      const tag = event.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || event.target?.isContentEditable) {
        return;
      }

      const key = (event.key || "").toLowerCase();
      if (key === "s") {
        event.preventDefault();
        stopSpeech();
        setActiveSpeakingId(null);
        return;
      }

      if (key === "r") {
        const latestAssistant = [...messages].reverse().find((m) => m.role === "assistant" && m.content);
        if (!latestAssistant) return;
        event.preventDefault();
        handleToggleSpeech(latestAssistant.id, latestAssistant.content);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeSection, handleToggleSpeech, messages]);

  const [voiceStatus, setVoiceStatus] = useState("");
  const [voiceAvailability, setVoiceAvailability] = useState({ en: false, hi: false, te: false, ta: false, kn: false, ml: false });

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const recognitionRef = useRef(null);
  const lastTranscriptRef = useRef("");
  const audioStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const [frequencyBars, setFrequencyBars] = useState(Array(12).fill(4));
  const micSupported = typeof window !== "undefined" && (("SpeechRecognition" in window) || ("webkitSpeechRecognition" in window));

  useEffect(() => {
    setMessages(prev => {
      if (prev.length === 0) {
        return [{
          id: "greeting", role: "assistant",
          content: tr("greeting"),
          sources: [], timestamp: new Date().toISOString()
        }];
      }
      // Update the content of the existing greeting message to match the selected language
      return prev.map(m => (m.id === "greeting" || m.id === "g") ? { ...m, content: tr("greeting") } : m);
    });
  }, [tr]);

  useEffect(() => {
    localStorage.setItem("chat_history", JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    localStorage.setItem("chat_session_id", sessionId);
  }, [sessionId]);

  const clearHistory = useCallback(() => {
    const greeting = {
      id: "greeting", role: "assistant",
      content: tr("greeting"),
      sources: [], timestamp: new Date().toISOString()
    };
    setMessages([greeting]);
    const nextSession = createSessionId();
    setSessionId(nextSession);
    setBackendHistory([]);
    setHistoryError("");
    localStorage.removeItem("chat_history");
  }, [createSessionId, tr]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (activeSection === "schemes" && schemes.length === 0) {
      getSchemes().then(d => setSchemes(d.schemes || [])).catch(() => {});
    }
    if (activeSection === "rights" && rights.length === 0) {
      getRights().then(d => setRights(d.rights || [])).catch(() => {});
    }
    if (activeSection === "assistive" && assistive.length === 0) {
      getAssistive().then(d => setAssistive(d.assistive || [])).catch(() => {});
    }
    if (activeSection === "accessibility" && accessibility.length === 0) {
      getAccessibility().then(d => setAccessibility(d.accessibility || [])).catch(() => {});
    }
    if (activeSection === "employment" && employment.length === 0) {
      getEmployment().then(d => setEmployment(d.employment || [])).catch(() => {});
    }
    if (activeSection === "education" && education.length === 0) {
      getEducation().then(d => setEducation(d.education || [])).catch(() => {});
    }
  }, [activeSection, schemes.length, rights.length, assistive.length, accessibility.length, employment.length, education.length, getSchemes, getRights, getAssistive, getAccessibility, getEmployment, getEducation]);

  useEffect(() => {
    if (activeSection !== "education") {
      setSelectedEducation(null);
    }
  }, [activeSection]);

  const loadBackendHistory = useCallback(async () => {
    if (!user) {
      setHistoryError("Login required to load backend chat history.");
      setBackendHistory([]);
      return;
    }

    setHistoryLoading(true);
    setHistoryError("");
    try {
      const rows = await getChatHistory(sessionId);
      setBackendHistory(rows);
    } catch (err) {
      setBackendHistory([]);
      setHistoryError("Unable to load chat history from backend.");
    } finally {
      setHistoryLoading(false);
    }
  }, [getChatHistory, sessionId, user]);

  const handleDeleteHistory = useCallback(() => {
    const ok = window.confirm("Delete this chat history?");
    if (!ok) return;
    clearHistory();
    setBackendHistory([]);
    setHistoryError("");
  }, [clearHistory]);

  useEffect(() => {
    if (activeSection === "history") {
      loadBackendHistory();
    }
  }, [activeSection, loadBackendHistory]);

  const sendMessage = useCallback(async (text, backendText) => {
    const displayText = (text || input).trim();
    const question = (backendText || text || input).trim();
    if (!displayText || !question || loading) return;
    setInput("");

    const userMsg = {
      id: Date.now(), role: "user",
      content: displayText, timestamp: new Date().toISOString()
    };
    setMessages(m => [...m, userMsg]);
    setLoading(true);

    try {
      const result = await chatQuery(question, sessionId, currentLang);
      if (result?.session_id && result.session_id !== sessionId) {
        setSessionId(result.session_id);
      }
      const aiMsg = {
        id: Date.now() + 1, role: "assistant",
        content: result.answer,
        sources: result.sources || [],
        timestamp: new Date().toISOString()
      };
      setMessages(m => [...m, aiMsg]);
    } catch (err) {
      setMessages(m => [...m, {
        id: Date.now() + 1, role: "assistant",
        content: tr("assistantUnavailable"),
        sources: [], timestamp: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.blur();
    }
  }, [input, loading, chatQuery, sessionId, currentLang, tr]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const stopVoiceInput = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
    }

    analyserRef.current = null;
    setFrequencyBars(Array(12).fill(4));
    setIsListening(false);
  }, []);

  const startVoiceInput = useCallback(async () => {
    if (!micSupported || loading) return;
    if (isListening) {
      stopVoiceInput();
      return;
    }

    const isLocalHost = ["localhost", "127.0.0.1"].includes(window.location.hostname);
    if (!window.isSecureContext && !isLocalHost) {
      setVoiceStatus("Voice input requires HTTPS or localhost.");
      return;
    }

    if (navigator.mediaDevices?.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioStreamRef.current = stream;

        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) {
          const audioContext = new AudioCtx();
          if (audioContext.state === "suspended") {
            await audioContext.resume();
          }
          const source = audioContext.createMediaStreamSource(stream);
          const analyser = audioContext.createAnalyser();
          analyser.fftSize = 256;
          analyser.smoothingTimeConstant = 0.8;
          source.connect(analyser);
          audioContextRef.current = audioContext;
          analyserRef.current = analyser;

          const freqData = new Uint8Array(analyser.frequencyBinCount);
          const timeData = new Uint8Array(analyser.fftSize);
          const renderFrequency = () => {
            if (!analyserRef.current) return;

            analyserRef.current.getByteFrequencyData(freqData);
            analyserRef.current.getByteTimeDomainData(timeData);

            const rms = Math.sqrt(
              timeData.reduce((sum, value) => {
                const centered = (value - 128) / 128;
                return sum + centered * centered;
              }, 0) / timeData.length
            );

            const levelBoost = Math.min(1, rms * 3.2);
            const step = Math.max(1, Math.floor(freqData.length / 12));
            const bars = Array.from({ length: 12 }, (_, index) => {
              const start = index * step;
              const slice = freqData.slice(start, Math.min(freqData.length, start + step));
              const avg = slice.length ? slice.reduce((sum, v) => sum + v, 0) / slice.length : 0;
              const frequencyLevel = avg / 255;
              const mixedLevel = Math.max(frequencyLevel, levelBoost * 0.9);
              return Math.max(4, Math.min(34, Math.round(4 + mixedLevel * 30)));
            });

            setFrequencyBars(bars);
            animationFrameRef.current = requestAnimationFrame(renderFrequency);
          };

          animationFrameRef.current = requestAnimationFrame(renderFrequency);
        }
      } catch (err) {
        setVoiceStatus("Microphone permission is blocked. Please allow mic access in the browser.");
        return;
      }
    }

    const RecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!RecognitionCtor) {
      setVoiceStatus(tr("voiceUnsupported"));
      return;
    }

    lastTranscriptRef.current = "";

    const recognition = new RecognitionCtor();
    recognition.lang = getSpeechLocale(currentLang);
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.continuous = true;

    let finalText = "";

    recognition.onstart = () => {
      setIsListening(true);
      setVoiceStatus(tr("voiceListening"));
    };

    recognition.onresult = (event) => {
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const chunk = event.results[i][0]?.transcript || "";
        if (event.results[i].isFinal) {
          finalText += `${chunk} `;
        } else {
          interimText += chunk;
        }
      }

      const preview = (finalText + interimText).trim();
      if (preview) {
        lastTranscriptRef.current = preview;
        setInput(preview);
      }
    };

    recognition.onerror = (event) => {
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        setVoiceStatus("Microphone permission is blocked. Please allow mic access in the browser.");
      } else if (event.error !== "aborted" && event.error !== "no-speech") {
        setVoiceStatus(tr("voiceUnsupported"));
      } else if (event.error === "no-speech") {
        setVoiceStatus("No speech detected. Please try again.");
      }
      setIsListening(false);
    };

    recognition.onend = () => {
      const transcript = (lastTranscriptRef.current || finalText || "").trim();
      if (transcript) {
        setInput(transcript);
        setTimeout(() => inputRef.current?.focus(), 50);
        setVoiceStatus("Voice captured. You can review and send.");
      } else {
        setVoiceStatus("No voice captured. Tap Voice and speak clearly.");
      }
      recognitionRef.current = null;
      stopVoiceInput();
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch (err) {
      setVoiceStatus("Unable to start voice input. Please tap Voice again.");
      setIsListening(false);
      recognitionRef.current = null;
    }
  }, [currentLang, isListening, loading, micSupported, stopVoiceInput, tr]);

  useEffect(() => {
    if (!("speechSynthesis" in window)) return;
    const preloadVoices = () => {
      window.speechSynthesis.getVoices();
      setVoiceAvailability(getVoiceAvailability());
    };
    preloadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", preloadVoices);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", preloadVoices);
  }, []);

  const missingIndianVoices = ["te", "ta", "kn", "ml"].filter((code) => !voiceAvailability[code]);

  useEffect(() => () => {
    stopVoiceInput();
  }, [stopVoiceInput]);

  const navItems = [
    { id: "chat", icon: <MessageSquare size={18}/>, label: tr("chat") },
    { id: "schemes", icon: <BookOpen size={18}/>, label: tr("schemes") },
    { id: "rights", icon: <Scale size={18}/>, label: tr("rights") },
    { id: "assistive", icon: <Award size={18}/>, label: tr("assistiveTech") },
    { id: "accessibility", icon: <Accessibility size={18}/>, label: tr("accessibility") },
    { id: "employment", icon: <Briefcase size={18}/>, label: tr("employment") },
    { id: "education", icon: <GraduationCap size={18}/>, label: tr("education") },
  ];

  const langObj = INDIAN_LANGUAGES.find(l => l.code === currentLang);
  const quickTopics = QUICK_TOPICS[currentLang] || QUICK_TOPICS.en;

  return (
    <div className="portal-shell">
      <header className="portal-header">
        <div className="container header-inner">
          <div className="header-left">
            <div className="brand" onClick={() => navigate("/")}>
              <span className="emblem" aria-hidden="true">♿</span>
              <div className="brand-text">
                <div className="govt-title">Disability Rights</div>
                <div className="govt-subtitle">{tr("portalSubtitle")}</div>
              </div>
            </div>
          </div>

          <div className="header-right">
            <label className="header-chip header-chip-select" aria-label="Language">
              <Globe size={14} aria-hidden="true" />
              <select value={currentLang} onChange={(e) => selectLanguage(e.target.value)}>
                {INDIAN_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>{lang.english}</option>
                ))}
              </select>
            </label>

            <button
              className="header-chip"
              onClick={() => {
                if (screenReaderEnabled) {
                  window.speechSynthesis.cancel();
                }
                toggleScreenReader();
              }}
              aria-pressed={screenReaderEnabled}
            >
              {screenReaderEnabled ? `🔊 ${tr("screenReaderOn")}` : `🔇 ${tr("screenReaderOff")}`}
            </button>

            <div className="user-profile">
              {user ? (
                <>
                  <span className="username" data-no-sr="true" aria-hidden="true">{user.full_name}</span>
                  <button 
                    className="logout-btn" 
                    onClick={() => { logout(); navigate("/"); }}
                  >
                    {tr("logout")}
                  </button>
                </>
              ) : (
                <button 
                  className="login-pill" 
                  onClick={() => navigate("/auth")}
                >
                  {tr("loginBtn")}
                </button>
              )}
            </div>
          </div>
        </div>
        <div className="tricolor-bar" aria-hidden="true">
          <div className="saffron" />
          <div className="white" />
          <div className="green" />
        </div>
      </header>

      <div className="portal-body">
        <aside className={`portal-sidebar ${sidebarOpen ? "open" : "closed"}`}>
          <div className="sidebar-scroll">
            <button 
              className="new-chat-btn" 
              onClick={() => { 
                clearHistory();
                setActiveSection("chat"); 
              }}
            >
              <Plus size={18} aria-hidden="true" />
              <span>+ {tr("newChat")}</span>
            </button>

            <button
              className="new-chat-btn"
              onClick={() => setActiveSection("history")}
            >
              <History size={18} aria-hidden="true" />
              <span>{tr("chatHistory")}</span>
            </button>


            <nav className="nav-list">
              {navItems.map(item => (
                <button
                  key={item.id}
                  className={`nav-link ${activeSection === item.id ? "active" : ""}`}
                  onClick={() => setActiveSection(item.id)}
                >
                  <span className="nav-icon" aria-hidden="true">{item.icon}</span>
                  <span className="nav-label">{item.label}</span>
                  {activeSection === item.id && <motion.div layoutId="nav-line" className="nav-active-line" />}
                </button>
              ))}
            </nav>

            <div className="sidebar-footer">
              <p className="footer-label">{tr("officialPortals")}</p>
              {OFFICIAL_PORTALS.map((portal) => (
                <a key={portal.labelKey} href={portal.url} target="_blank" rel="noreferrer">
                  <ExternalLink size={14} /> {tr(portal.labelKey)}
                </a>
              ))}
            </div>
          </div>
        </aside>

        <main className="portal-main">
          <AnimatePresence mode="wait">
            {activeSection === "chat" ? (
              <motion.div 
                key="chat"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="workspace-chat"
              >

                
                <div className="messages-area">
                  {messages.map(msg => (
                    <Message 
                      key={msg.id} 
                      msg={msg} 
                      tr={tr} 
                      activeSpeakingId={activeSpeakingId}
                      onToggleSpeech={handleToggleSpeech}
                    />
                  ))}
                  {loading && <TypingIndicator />}
                  <div ref={messagesEndRef} />
                </div>

                <div className="suggested-prompts">
                  <p>{tr("quickTopics")}: {tr("clickToAsk")}</p>
                  <div className="prompt-grid">
                    {quickTopics.map((topic) => (
                      <button key={topic.label} onClick={() => sendMessage(topic.label, topic.query)} className="prompt-pill">
                        {topic.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="chat-input-container">
                  <div className="chat-input-row">
                    <button
                      className={`voice-side-btn ${isListening ? "listening" : ""}`}
                      onClick={startVoiceInput}
                      disabled={!micSupported || loading}
                      title={isListening ? tr("voiceInputStop") : tr("voiceInputStart")}
                      aria-label={isListening ? tr("voiceInputStop") : tr("voiceInputStart")}
                      aria-pressed={isListening}
                    >
                      {isListening ? <MicOff size={18} aria-hidden="true" /> : <Mic size={18} aria-hidden="true" />}
                      <span>Voice</span>
                    </button>
                    {isListening && (
                      <div className="voice-frequency" aria-label="Microphone input level" aria-hidden="true">
                        {frequencyBars.map((height, index) => (
                          <span
                            key={`bar_${index}`}
                            className="voice-frequency-bar"
                            style={{ height: `${height}px` }}
                          />
                        ))}
                      </div>
                    )}
                    <div className="input-box">
                      <textarea 
                        ref={inputRef}
                        placeholder={tr("askQuestion")}
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        rows={1}
                      />
                      <button 
                        className="send-btn" 
                        onClick={() => sendMessage()}
                        disabled={loading || !input.trim()}
                        aria-label={tr("send")}
                      >
                        <Send size={18} aria-hidden="true" />
                      </button>
                    </div>
                  </div>
                  <div className="input-features">
                    <span><Keyboard size={14} aria-hidden="true" /> {tr("chatKeyboardHint")}</span>
                    <span><Keyboard size={14} aria-hidden="true" /> Commands: R = Read, S = Stop</span>
                    <span><Globe size={14} aria-hidden="true" /> {tr("responsesIn")} {langObj?.english}</span>
                    {voiceStatus && <span><Mic size={14} aria-hidden="true" /> {voiceStatus}</span>}
                    {missingIndianVoices.length > 0 && (
                      <span>
                        <Mic size={14} />
                        Missing voices: {missingIndianVoices.join(", ").toUpperCase()} (install in Windows Language settings)
                      </span>
                    )}
                  </div>
                </div>
              </motion.div>
            ) : activeSection === "history" ? (
              <motion.div
                key="history"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="workspace-info"
              >
                <div className="info-header">
                  <h1>{tr("chatHistory")}</h1>
                  <p>Session ID: {sessionId}</p>
                </div>

                <div className="suggested-prompts history-actions" style={{ marginBottom: 12 }}>
                  <button className="prompt-pill" onClick={loadBackendHistory} disabled={historyLoading}>
                    {historyLoading ? "Loading..." : "Refresh History"}
                  </button>
                  <button className="prompt-pill history-delete-btn" onClick={handleDeleteHistory}>
                    Delete Chat
                  </button>
                </div>

                {!user && <p>Please login to view backend chat history.</p>}
                {historyError && <p>{historyError}</p>}
                {!historyLoading && user && !historyError && backendHistory.length === 0 && (
                  <p>No backend history found for this session yet.</p>
                )}

                <div className="history-thread">
                  {backendHistory.map((item, index) => {
                    if (item.role !== "user") return null;
                    const next = backendHistory[index + 1];
                    const assistantReply = next && next.role === "assistant" ? next : null;

                    return (
                      <div className="history-thread-item" key={`history_pair_${index}`}>
                        <div className="history-bubble user">
                          <p>{item.content}</p>
                        </div>
                        {assistantReply && (
                          <div className="history-bubble assistant">
                            <p>{assistantReply.content}</p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </motion.div>
            ) : (
              <motion.div 
                key={activeSection}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="workspace-info"
              >
                <div className="info-header">
                  <h1>{navItems.find(i => i.id === activeSection)?.label}</h1>
                  <p>{tr("infoHeaderSubtitle")}</p>
                </div>

                {activeSection === "schemes" && (
                  <div className="info-grid">
                    {schemes.map(s => (
                      <div key={s.id} className="resource-card premium-card" data-category="schemes">
                        <div className="card-badge">{tr("badgeScheme")}</div>
                        <h3>{s.name}</h3>
                        <p className="ministry">{s.ministry}</p>
                        <p className="desc">{s.desc}</p>
                        <a href={s.url} target="_blank" rel="noreferrer">{tr("visitSite")} <ExternalLink size={14}/></a>
                      </div>
                    ))}
                  </div>
                )}

                {activeSection === "rights" && (
                   <div className="info-grid">
                    {rights.map((r, i) => (
                      <div key={i} className="resource-card premium-card right" data-category="rights">
                        <div className="card-badge secondary">{r.section}</div>
                        <h3>{r.title}</h3>
                        <p className="desc">{r.desc}</p>
                      </div>
                    ))}
                  </div>
                )}

                {activeSection === "assistive" && (
                  <div className="info-grid">
                    {assistive.map(a => (
                      <div key={a.id} className="resource-card premium-card" data-category="assistive">
                        <div className="card-badge">{a.category}</div>
                        <h3>{a.name}</h3>
                        <p className="desc">{a.desc}</p>
                      </div>
                    ))}
                  </div>
                )}

                {activeSection === "accessibility" && (
                  <div className="info-grid">
                    {accessibility.map((a, i) => (
                      <div key={i} className="resource-card premium-card" data-category="accessibility">
                        <div className="card-badge">{tr("badgeStandard")}</div>
                        <h3>{a.title}</h3>
                        <p className="desc">{a.desc}</p>
                      </div>
                    ))}
                  </div>
                )}

                {activeSection === "employment" && (
                  <div className="info-grid">
                    {employment.map((e, i) => (
                      <div key={i} className="resource-card premium-card" data-category="employment">
                        <div className="card-badge">{tr("badgeEmployment")}</div>
                        <h3>{e.title}</h3>
                        <p className="desc">{e.desc}</p>
                      </div>
                    ))}
                  </div>
                )}

                {activeSection === "education" && (
                  <>
                    <div className="info-grid">
                      {education.map((e, i) => {
                        const educationId = e.id || `education_${i}`;
                        const isSelected = selectedEducation?.id === educationId;
                        return (
                          <button
                            key={educationId}
                            type="button"
                            className={`resource-card resource-card-button premium-card ${isSelected ? "selected" : ""}`}
                            data-category="education"
                            onClick={() => setSelectedEducation({ ...e, id: educationId })}
                            aria-pressed={isSelected}
                          >
                            <div className="card-badge">{tr("badgeEducation")}</div>
                            <h3>{e.title}</h3>
                            <p className="desc">{e.desc}</p>
                          </button>
                        );
                      })}
                    </div>

                    {selectedEducation && (
                      <div className="education-detail-panel" role="status" aria-live="polite">
                        <h3>{selectedEducation.title}</h3>
                        <p>{selectedEducation.details || selectedEducation.desc}</p>
                      </div>
                    )}
                  </>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
