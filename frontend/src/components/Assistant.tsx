"use client";
import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { API_URL, apiError } from "@/lib/config";
import { Chip, toneForRisk } from "./Bento";

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
 * Grounded Q&A over the current result. Calls go through our own backend, never
 * to the provider: a static site cannot hold an API key. See backend/llm.py.
 */
export default function Assistant({ result, pending }: {
    result: any;
    // Bumped by the report's "Ask CliniViz" buttons; `n` makes the same
    // question fire again rather than being swallowed as an unchanged prop.
    pending?: { q: string; n: number } | null;
}) {
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

    // A new case resets the thread: old turns would misdescribe the new report.
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
            setError(apiError(err, "CliniViz"));
        } finally {
            setBusy(false);
        }
    };

    useEffect(() => {
        if (pending?.q) ask(pending.q);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pending?.n]);

    if (status && !status.configured) {
        return (
            <div className="panel-inset rounded-2xl p-6 max-w-xl">
                <h2 className="text-sm font-black uppercase tracking-[0.15em] text-brand mb-3">
                    CliniViz is not configured
                </h2>
                <p className="text-xs text-muted leading-relaxed mb-3">
                    It runs through the backend so the API key is never shipped to the
                    browser. Set these on the server and restart:
                </p>
                <pre className="text-[10px] panel-inset rounded-lg p-3 overflow-x-auto text-muted">
{`LLM_PROVIDER=${status.provider}      # groq | grok | nemotron
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
    const tone = toneForRisk(result?.risk);

    return (
        <div className="grid lg:grid-cols-[260px_minmax(0,1fr)] gap-4 items-start">
            {/* Context rail: what the assistant can actually see. */}
            <aside className="space-y-3">
                <div className="panel-inset rounded-2xl p-4">
                    <p className="text-[9px] font-black text-subtle uppercase tracking-[0.15em] mb-2">Grounded on</p>
                    {result ? (
                        <>
                            <div className="flex items-center gap-2 mb-2.5">
                                <Chip tone={tone}>{result.risk}</Chip>
                                <span className="text-[11px] font-bold truncate">{result._fileName || "current scan"}</span>
                            </div>
                            <dl className="space-y-1 text-[11px]">
                                {[
                                    ["Fused risk", `${(result.risk_probability * 100).toFixed(1)}%`],
                                    ["P(ulcer)", `${(result.image_probability * 100).toFixed(1)}%`],
                                    ["IWGDF", result.iwgdf?.category ?? "—"],
                                    ["Clinical shift", result.clinical_logit_shift?.toFixed(2)],
                                ].map(([k, v]) => (
                                    <div key={String(k)} className="flex justify-between gap-2">
                                        <dt className="text-subtle">{k}</dt>
                                        <dd className="font-bold tabular-nums">{v}</dd>
                                    </div>
                                ))}
                            </dl>
                        </>
                    ) : (
                        <p className="text-[11px] text-subtle leading-relaxed">
                            No scan analysed yet. You can still ask how the pipeline works —
                            CliniViz just has no patient result to quote.
                        </p>
                    )}
                </div>

                {suggestions.length > 0 && (
                    <div className="panel-inset rounded-2xl p-4">
                        <p className="text-[9px] font-black text-subtle uppercase tracking-[0.15em] mb-2">Try asking</p>
                        <div className="space-y-1.5">
                            {suggestions.map((s) => (
                                <button key={s} type="button" onClick={() => ask(s)} disabled={busy}
                                    className="w-full text-left text-[11px] leading-snug px-2.5 py-2 rounded-lg text-muted hover:text-fg hover:bg-brand/10 transition-colors disabled:opacity-50">
                                    {s}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {status && (
                    <p className="text-[10px] text-subtle leading-relaxed px-1">
                        {status.provider} · {status.model}. Answers come only from the report
                        JSON; it says so when something is not in there rather than guessing.
                        Not a substitute for a clinician.
                    </p>
                )}
            </aside>

            {/* Transcript. */}
            <div className="panel-inset rounded-2xl p-4 flex flex-col min-h-[26rem]">
                <div ref={scrollRef} className="flex-1 overflow-y-auto pr-1 space-y-3 max-h-[calc(100vh-22rem)]">
                    {turns.length === 0 && !busy && (
                        <div className="h-full grid place-items-center text-center py-10">
                            <div>
                                <div className="w-11 h-11 rounded-xl neon bg-surface2/60 grid place-items-center mx-auto mb-3">
                                    <svg className="w-5 h-5 text-brand" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.6} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.86 9.86 0 01-4-.8L3 20l1.2-3.2A7.5 7.5 0 013 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                                    </svg>
                                </div>
                                <p className="text-sm font-bold">Ask CliniViz</p>
                                <p className="text-[11px] text-subtle mt-1 max-w-xs mx-auto leading-relaxed">
                                    Pick a question on the left, or type your own below.
                                </p>
                            </div>
                        </div>
                    )}

                    {turns.map((t, i) => (
                        <div key={i}
                            className={`text-sm leading-relaxed rounded-xl px-4 py-3 animate-fadeUp max-w-[46rem] ${t.role === "user"
                                ? "bg-brand/10 border border-brand/20 ml-auto"
                                : "bg-surface2/70 border border-line/10"}`}>
                            <p className="text-[9px] font-black uppercase tracking-[0.15em] text-subtle mb-1">
                                {t.role === "user" ? "You" : "CliniViz"}
                            </p>
                            <p className="whitespace-pre-wrap">{t.content}</p>
                        </div>
                    ))}

                    {busy && (
                        <div className="bg-surface2/70 border border-line/10 rounded-xl px-4 py-3 flex items-center gap-2 text-xs text-muted w-fit">
                            <svg className="animate-spin h-4 w-4 text-brand" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                            Thinking…
                        </div>
                    )}
                </div>

                {error && <p role="alert" className="text-xs text-danger mt-3 leading-relaxed">{error}</p>}

                <form className="flex gap-2 mt-3 pt-3 border-t border-line/10"
                    onSubmit={(e) => { e.preventDefault(); ask(q); }}>
                    <input
                        className="field flex-1 py-2.5 text-sm"
                        value={q}
                        onChange={(e) => setQ(e.target.value)}
                        placeholder={result ? "Ask CliniViz about this result…" : "Ask CliniViz how the system works…"}
                        maxLength={1000}
                        disabled={busy}
                        aria-label="Question for CliniViz"
                    />
                    <button type="submit" disabled={busy || !q.trim()}
                        className="px-5 rounded-xl bg-brand text-brandFg font-black text-[11px] uppercase tracking-[0.12em] disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition-opacity">
                        Ask
                    </button>
                </form>
            </div>
        </div>
    );
}
