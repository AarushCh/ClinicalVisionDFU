"use client";
import { useState, useCallback } from "react";
import UploadForm from "@/components/UploadForm";
import ResultCard from "@/components/ResultCard";
import SampleGallery from "@/components/SampleGallery";
import Assistant from "@/components/Assistant";
import ThemeToggle from "@/components/ThemeToggle";
import Ambient from "@/components/Ambient";
import HistoryPanel from "@/components/HistoryPanel";
import { Preset } from "@/lib/config";
import { addHistory, makeThumb, toEntry } from "@/lib/history";
import { modelLabel, useModelInfo } from "@/lib/model";

type Tab = "report" | "assistant" | "history";

const TABS: { id: Tab; label: string; icon: string }[] = [
    { id: "report", label: "Report", icon: "M9 17v-6h6v6m2 4H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z" },
    { id: "assistant", label: "CliniViz", icon: "M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.86 9.86 0 01-4-.8L3 20l1.2-3.2A7.5 7.5 0 013 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" },
    { id: "history", label: "History", icon: "M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" },
];

export default function Home() {
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [pending, setPending] = useState<{ preset: Preset; file: File } | null>(null);
    const [historyVersion, setHistoryVersion] = useState(0);
    const [tab, setTab] = useState<Tab>("report");
    const [pendingQuestion, setPendingQuestion] = useState<{ q: string; n: number } | null>(null);

    // The report's "Ask CliniViz" buttons: switch tab and hand over the question.
    const askCliniViz = useCallback((q: string) => {
        setPendingQuestion({ q, n: Date.now() });
        setTab("assistant");
    }, []);
    const model = useModelInfo();

    // Fire-and-forget: a storage failure must not break the result on screen.
    const handleResult = useCallback(async (r: any) => {
        setResult(r);
        if (!r) return;
        setTab("report");
        try {
            const thumb = await makeThumb(
                r._previewUrl || `data:image/png;base64,${r.overlay || r.heatmap}`
            );
            await addHistory(toEntry(r, r._fileName || "scan", thumb));
            setHistoryVersion((v) => v + 1);
        } catch {
            // ignore: the report on screen is unaffected
        }
    }, []);

    const handlePick = useCallback((preset: Preset, file: File) => {
        setPending({ preset, file });
        setResult(null);
        setTab("report");
    }, []);

    return (
        <main className="min-h-screen bg-bg text-fg selection:bg-brand/30 relative flex flex-col">
            <Ambient />

            <header className="sticky top-0 z-30 border-b border-line/10 bg-bg/85 backdrop-blur-xl">
                <div className="mx-auto max-w-[1920px] px-4 lg:px-6 h-14 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-2.5 min-w-0">
                        <div className="p-1.5 bg-brand/15 rounded-lg border border-brand/25 text-brand shrink-0">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                            </svg>
                        </div>
                        <p className="font-display leading-none truncate">ClinicalVision <span className="text-brand">DFU</span></p>
                        <span className="hidden md:inline-flex items-center gap-1.5 ml-1 px-2 py-0.5 rounded-full bg-good/10 border border-good/25 text-good text-[9px] font-black uppercase tracking-[0.15em]">
                            <span className="w-1.5 h-1.5 rounded-full bg-good animate-pulse" />
                            Live
                        </span>
                    </div>

                    <div className="flex items-center gap-3 shrink-0">
                        <span className="hidden lg:block text-[10px] text-subtle uppercase tracking-[0.12em]">
                            {modelLabel(model)}
                        </span>
                        <ThemeToggle />
                    </div>
                </div>
            </header>

            <div className="flex-1 mx-auto w-full max-w-[1920px] px-4 lg:px-6 py-4 relative z-10">
                {/* Collapsed by default; the full text still has to be reachable. */}
                <details className="group mb-4 rounded-xl bg-warn/10 border border-warn/25">
                    <summary className="flex items-center gap-2.5 px-3.5 py-2 cursor-pointer select-none list-none">
                        <svg className="w-3.5 h-3.5 text-warn shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M5.07 19h13.86a2 2 0 001.71-3.03l-6.93-12a2 2 0 00-3.42 0l-6.93 12A2 2 0 005.07 19z" />
                        </svg>
                        <p className="text-[11px] text-warn/90 leading-snug min-w-0">
                            <strong>Research prototype — not a medical device.</strong>
                            <span className="hidden sm:inline"> Not FDA/CE cleared; never use in place of a qualified clinician.</span>
                            <span className="text-warn/60"> Full disclaimer →</span>
                        </p>
                    </summary>
                    <p className="text-[11px] text-warn/90 leading-relaxed px-3.5 pb-3 pt-0.5 pl-[2.4rem]">
                        Not validated against clinical outcomes. Clinical risk coefficients are
                        literature-informed priors, not fitted to patient data. The model accepts
                        only close-cropped tissue images; anything else — a whole-foot photograph
                        included — is classified as an ulcer with high confidence and its result
                        should be disregarded. Do not upload identifiable patient data.
                    </p>
                </details>

                <div className="grid grid-cols-1 xl:grid-cols-[minmax(290px,320px)_minmax(0,1fr)] gap-4 items-start">
                    <div className="space-y-4 xl:sticky xl:top-[4.5rem]">
                        <UploadForm
                            setResult={handleResult}
                            setLoading={setLoading}
                            loading={loading}
                            pending={pending}
                            clearPending={() => setPending(null)}
                        />
                        <SampleGallery onPick={handlePick} disabled={loading} />
                    </div>

                    <div className="min-w-0">
                        {/* Tabs, not a stack: both used to be two screens down. */}
                        <div className="flex items-center justify-between gap-3 mb-3">
                            <div className="flex gap-1 panel p-1 rounded-xl">
                                {TABS.map((t) => (
                                    <button key={t.id} onClick={() => setTab(t.id)}
                                        aria-current={tab === t.id}
                                        className={`flex items-center gap-1.5 px-3 sm:px-4 py-2 rounded-lg text-[10px] font-black uppercase tracking-[0.12em] transition-colors ${tab === t.id ? "bg-brand/20 text-brand" : "text-subtle hover:text-fg"}`}>
                                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={t.icon} />
                                        </svg>
                                        {t.label}
                                    </button>
                                ))}
                            </div>
                            <p className="hidden lg:block text-[10px] text-subtle uppercase tracking-[0.12em] truncate">
                                {result
                                    ? `${modelLabel(result.model)} · ${result.inference_ms?.toFixed(0)} ms · ${result.cam_mode}`
                                    : `${modelLabel(model)} · Grad-CAM++ · IWGDF 2023`}
                            </p>
                        </div>

                        {/* Kept mounted so a tab switch keeps the transcript. */}
                        <div hidden={tab !== "report"}>
                            <ResultCard result={result} loading={loading} onAsk={askCliniViz} />
                        </div>
                        <div hidden={tab !== "assistant"}>
                            <Assistant result={result} pending={pendingQuestion} />
                        </div>
                        <div hidden={tab !== "history"}>
                            <HistoryPanel version={historyVersion} />
                        </div>
                    </div>
                </div>
            </div>

            <footer className="mx-auto w-full max-w-[1920px] px-4 lg:px-6 py-4 mt-2 border-t border-line/10 text-[10px] text-subtle flex flex-wrap gap-x-5 gap-y-1.5 justify-between relative z-10">
                <span>{modelLabel(model)} · Grad-CAM++ · log-odds fusion · IWGDF 2023</span>
                <a className="hover:text-fg transition-colors"
                    href="https://github.com/aarushch/ClinicalVisionDFU"
                    target="_blank" rel="noreferrer">
                    Source &amp; technical report →
                </a>
            </footer>
        </main>
    );
}
