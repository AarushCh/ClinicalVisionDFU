"use client";
import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { API_URL } from "@/lib/config";

type Turn = { role: "user" | "assistant"; content: string };
type Status = {
    provider: string;
    model: string;
    configured: boolean;
    key_env: string;
    docs: string;
    suggested_questions: string[];
};

/**
 * Grounded Q&A over the current result.
 *
 * Every call goes to our own backend, never to the model provider directly: a
 * static site cannot hold an API key without shipping it to every visitor. The
 * backend attaches the key and constrains the model to answer only from the
 * report JSON (see backend/llm.py).
 */
export default function Assistant({ result }: { result: any }) {
    const [status, setStatus] = useState<Status | null>(null);
    const [turns, setTurns] = useState<Turn[]>([]);
    const [q, setQ] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        axios
            .get(`${API_URL}/assistant`, { timeout: 15000 })
            .then((r) => setStatus(r.data))
            .catch(() => setStatus(null));
    }, []);

    // Reset the conversation when a new case is analysed — history about the
    // previous image would be actively misleading against a new report.
    useEffect(() => {
        setTurns([]);
        setError(null);
    }, [result?.inference_ms, result?.risk_probability]);

    useEffect(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }, [turns, busy]);

    const ask = async (question: string) => {
        const text = question.trim();
        if (!text || busy) return;
        setError(null);
        setQ("");
        const history = turns.slice(-6);
        setTurns((t) => [...t, { role: "user", content: text }]);
        setBusy(true);
        try {
            const r = await axios.post(
                `${API_URL}/assistant/ask`,
                { question: text, result, history },
                { timeout: 90000 }
            );
            setTurns((t) => [...t, { role: "assistant", content: r.data.answer }]);
        } catch (err: any) {
            const detail = err?.response?.data?.detail;
            setError(
                typeof detail === "string"
                    ? detail
                    : err?.code === "ECONNABORTED"
                        ? "The model took too long to respond."
                        : `Could not reach the assistant at ${API_URL}.`
            );
        } finally {
            setBusy(false);
        }
    };

    if (status && !status.configured) {
        return (
            <div className="panel rounded-[2rem] p-6">
                <h2 className="text-sm font-black uppercase tracking-[0.15em] text-brand mb-3">
                    AI Assistant
                </h2>
                <p className="text-xs text-muted leading-relaxed mb-3">
                    Not configured. The assistant runs through the backend so the API key
                    is never shipped to the browser. To enable it, set an environment
                    variable on the server and restart:
                </p>
                <pre className="text-[10px] panel-inset rounded-lg p-3 overflow-x-auto text-muted">
{`LLM_PROVIDER=${status.provider}      # grok | nemotron
LLM_API_KEY=<your key>`}
                </pre>
                <p className="text-[10px] text-subtle mt-3">
                    Free keys: <a className="text-brand hover:underline" href={status.docs} target="_blank" rel="noreferrer">{status.docs}</a>
                    {" · "}current model: <code>{status.model}</code>
                </p>
            </div>
        );
    }

    const suggestions = status?.suggested_questions ?? [];

    return (
        <div className="panel rounded-[2rem] p-6 flex flex-col">
            <div className="flex items-baseline justify-between gap-3 mb-1">
                <h2 className="text-sm font-black uppercase tracking-[0.15em] text-brand">
                    Ask about this result
                </h2>
                {status && (
                    <span className="text-[9px] text-subtle uppercase tracking-wider">
                        {status.provider} · {status.model}
                    </span>
                )}
            </div>
            <p className="text-xs text-muted mb-4 leading-relaxed">
                Answers are grounded strictly in the report above. The assistant will say
                so when something is not in the report rather than guessing.
            </p>

            {turns.length === 0 && (
                <div className="flex flex-wrap gap-2 mb-4">
                    {suggestions.map((s) => (
                        <button
                            key={s}
                            type="button"
                            onClick={() => ask(s)}
                            disabled={busy}
                            className="text-[11px] px-3 py-1.5 rounded-full panel-inset hover:border-brand/40 text-muted hover:text-fg transition-colors disabled:opacity-50"
                        >
                            {s}
                        </button>
                    ))}
                </div>
            )}

            {turns.length > 0 && (
                <div
                    ref={scrollRef}
                    className="space-y-3 max-h-[22rem] overflow-y-auto mb-4 pr-1"
                >
                    {turns.map((t, i) => (
                        <div
                            key={i}
                            className={`text-sm leading-relaxed rounded-xl px-4 py-3 animate-fadeUp ${t.role === "user"
                                ? "bg-brand/10 border border-brand/20 ml-6"
                                : "panel-inset mr-6"
                                }`}
                        >
                            <p className="text-[9px] font-black uppercase tracking-[0.15em] text-subtle mb-1">
                                {t.role === "user" ? "You" : "Assistant"}
                            </p>
                            <p className="whitespace-pre-wrap">{t.content}</p>
                        </div>
                    ))}
                    {busy && (
                        <div className="panel-inset rounded-xl px-4 py-3 mr-6 flex items-center gap-2 text-xs text-muted">
                            <svg className="animate-spin h-4 w-4 text-brand" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                            Thinking…
                        </div>
                    )}
                </div>
            )}

            {error && (
                <p role="alert" className="text-xs text-danger mb-3 leading-relaxed">{error}</p>
            )}

            <form
                className="flex gap-2 mt-auto"
                onSubmit={(e) => {
                    e.preventDefault();
                    ask(q);
                }}
            >
                <input
                    className="field flex-1"
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    placeholder={result ? "Ask about this result…" : "Ask how the system works…"}
                    maxLength={1000}
                    disabled={busy}
                    aria-label="Question for the assistant"
                />
                <button
                    type="submit"
                    disabled={busy || !q.trim()}
                    className="px-4 rounded-xl bg-brand text-brandFg font-bold text-xs uppercase tracking-wider disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
                >
                    Ask
                </button>
            </form>

            <p className="text-[10px] text-subtle mt-3 leading-relaxed">
                Language-model output about a research prototype. It cannot diagnose, and
                it is not a substitute for a qualified clinician.
            </p>
        </div>
    );
}
