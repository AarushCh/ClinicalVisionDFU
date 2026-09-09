"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import UploadForm from "@/components/UploadForm";
import ResultCard from "@/components/ResultCard";
import SampleGallery from "@/components/SampleGallery";
import Assistant from "@/components/Assistant";
import ThemeToggle from "@/components/ThemeToggle";
import AuthGate, { useSession } from "@/components/AuthGate";
import HistoryPanel from "@/components/HistoryPanel";
import { Preset } from "@/lib/config";
import { addHistory, makeThumb, toEntry, HistoryEntry } from "@/lib/history";

export default function Home() {
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [pending, setPending] = useState<{ preset: Preset; file: File } | null>(null);
    const [historyVersion, setHistoryVersion] = useState(0);
    const { session, ready, signIn, signOut } = useSession();

    // Persist every completed analysis. Deliberately fire-and-forget: history is
    // a convenience, and a storage failure must never break the result the user
    // is looking at.
    const handleResult = useCallback(async (r: any) => {
        setResult(r);
        if (!r) return;
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

    const [mouse, setMouse] = useState({ x: 0, y: 0 });
    const current = useRef({ x: 0, y: 0 });
    const target = useRef({ x: 0, y: 0 });

    useEffect(() => {
        // Skip the cursor-glow rAF loop entirely on touch devices and for users
        // who asked for reduced motion: it is decoration, and it costs a frame
        // callback for the life of the page.
        if (typeof window === "undefined") return;
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
        if (!window.matchMedia("(hover: hover)").matches) return;

        const onMove = (e: MouseEvent) => {
            target.current = { x: e.clientX, y: e.clientY };
        };
        window.addEventListener("mousemove", onMove, { passive: true });

        let id: number;
        const tick = () => {
            current.current.x += (target.current.x - current.current.x) * 0.08;
            current.current.y += (target.current.y - current.current.y) * 0.08;
            setMouse({ x: current.current.x, y: current.current.y });
            id = requestAnimationFrame(tick);
        };
        tick();
        return () => {
            window.removeEventListener("mousemove", onMove);
            cancelAnimationFrame(id);
        };
    }, []);

    const handlePick = useCallback((preset: Preset, file: File) => {
        setPending({ preset, file });
        setResult(null);
    }, []);

    // The session lives in localStorage, so it cannot be known during SSR. Show a
    // branded placeholder for that one frame rather than a blank white page.
    if (!ready) {
        return (
            <main className="min-h-screen bg-bg grid place-items-center">
                <div className="flex items-center gap-3 text-subtle animate-pulse">
                    <div className="p-2 bg-brand/15 rounded-lg border border-brand/25 text-brand">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                    </div>
                    <span className="text-sm font-bold tracking-tight">ClinicalVision DFU</span>
                </div>
            </main>
        );
    }
    if (!session) return <AuthGate onSignIn={signIn} />;

    return (
        <main className="min-h-screen bg-bg text-fg selection:bg-brand/30 relative">
            {/* Decorative layers. pointer-events-none so they never intercept clicks. */}
            <div
                className="pointer-events-none fixed inset-0 z-0 hidden lg:block"
                style={{
                    background: `radial-gradient(700px circle at ${mouse.x}px ${mouse.y}px, rgb(var(--brand) / 0.08), transparent 42%)`,
                }}
            />
            <div className="ambient fixed inset-0 pointer-events-none z-0 overflow-hidden">
                <div className="absolute top-[-20%] left-[-10%] w-[45%] h-[45%] bg-brand/10 blur-[150px] rounded-full" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[150px] rounded-full" />
                <div className="absolute top-[35%] right-[-12%] w-[35%] h-[35%] bg-purple-600/10 blur-[150px] rounded-full" />
            </div>

            <header className="sticky top-0 z-30 border-b border-line/10 bg-bg/80 backdrop-blur-xl">
                <div className="mx-auto max-w-[1700px] px-4 sm:px-8 h-16 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3 min-w-0">
                        <div className="p-2 bg-brand/15 rounded-lg border border-brand/25 text-brand shrink-0">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                            </svg>
                        </div>
                        <div className="min-w-0">
                            <p className="font-black tracking-tight leading-none truncate">ClinicalVision DFU</p>
                            <p className="text-[10px] text-subtle tracking-wide">Explainable AI triage</p>
                        </div>
                        <span className="hidden md:inline-flex items-center gap-1.5 ml-3 px-2.5 py-1 rounded-full bg-good/10 border border-good/25 text-good text-[9px] font-black uppercase tracking-[0.15em]">
                            <span className="w-1.5 h-1.5 rounded-full bg-good animate-pulse" />
                            System active
                        </span>
                    </div>

                    <div className="flex items-center gap-2 sm:gap-3 shrink-0">
                        <div className="hidden sm:block text-right leading-tight">
                            <p className="text-xs font-bold truncate max-w-[160px]">{session.name}</p>
                            <p className="text-[10px] text-subtle">{session.role}</p>
                        </div>
                        <ThemeToggle />
                        <button
                            onClick={signOut}
                            className="px-3 py-2 rounded-xl border border-line/20 bg-surface/5 hover:bg-surface/10 text-muted hover:text-fg transition-colors text-[10px] font-black uppercase tracking-[0.15em]"
                        >
                            Sign out
                        </button>
                    </div>
                </div>
            </header>

            <div className="mx-auto max-w-[1700px] px-4 sm:px-8 py-8 relative z-10">
                <div className="mb-7 flex items-start gap-3 px-5 py-3 rounded-2xl bg-warn/10 border border-warn/25">
                    <svg className="w-5 h-5 text-warn shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M5.07 19h13.86a2 2 0 001.71-3.03l-6.93-12a2 2 0 00-3.42 0l-6.93 12A2 2 0 005.07 19z" />
                    </svg>
                    <p className="text-xs text-warn/90 leading-relaxed">
                        <strong>Research prototype — not a medical device.</strong> Not
                        FDA/CE cleared and not validated against clinical outcomes. Clinical
                        risk coefficients are literature-informed priors, not fitted to
                        patient data. Never use in place of a qualified clinician, and do
                        not upload identifiable patient data.
                    </p>
                </div>

                {/* Fixed-width control rail + fluid report column. The previous
                    5/7 split inside a 1280px container left the report roughly
                    745px wide, which its own nested two-column grid then halved
                    again — narrow enough that body text wrapped every few words. */}
                <div className="grid grid-cols-1 xl:grid-cols-[minmax(340px,400px)_minmax(0,1fr)] gap-6 items-start">
                    <div className="space-y-6 xl:sticky xl:top-24">
                        <UploadForm
                            setResult={handleResult}
                            setLoading={setLoading}
                            loading={loading}
                            pending={pending}
                            clearPending={() => setPending(null)}
                        />
                        <SampleGallery onPick={handlePick} disabled={loading} />
                    </div>

                    <div className="space-y-6 min-w-0">
                        <ResultCard result={result} loading={loading} session={session} />
                        <Assistant result={result} />
                        <HistoryPanel version={historyVersion} />
                    </div>
                </div>

                <footer className="mt-12 pt-6 border-t border-line/10 text-[11px] text-subtle flex flex-wrap gap-x-6 gap-y-2 justify-between">
                    <span>
                        ResNet-18 @ 224px · Grad-CAM++ · log-odds clinical fusion · IWGDF 2023
                    </span>
                    <a
                        className="hover:text-fg transition-colors"
                        href="https://github.com/aarushch/ClinicalVisionDFU"
                        target="_blank"
                        rel="noreferrer"
                    >
                        Source &amp; technical report →
                    </a>
                </footer>
            </div>
        </main>
    );
}
