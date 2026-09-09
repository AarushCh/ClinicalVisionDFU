"use client";
import { useState, useEffect, useRef } from "react";
import UploadForm from "@/components/UploadForm";
import ResultCard from "@/components/ResultCard";

export default function Home() {
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const currentPos = useRef({ x: 0, y: 0 });
    const targetPos = useRef({ x: 0, y: 0 });

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            targetPos.current = { x: e.clientX, y: e.clientY };
        };
        window.addEventListener("mousemove", handleMouseMove);

        let animationFrameId: number;
        const updatePosition = () => {
            currentPos.current.x += (targetPos.current.x - currentPos.current.x) * 0.08;
            currentPos.current.y += (targetPos.current.y - currentPos.current.y) * 0.08;
            setMousePosition({ x: currentPos.current.x, y: currentPos.current.y });
            animationFrameId = requestAnimationFrame(updatePosition);
        };
        updatePosition();

        return () => {
            window.removeEventListener("mousemove", handleMouseMove);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <main className="min-h-screen bg-[#030712] text-slate-200 font-sans selection:bg-cyan-500/30 relative overflow-hidden">

            {/* 1. INTERACTIVE CURSOR GLOW */}
            <div
                className="pointer-events-none fixed inset-0 z-0 transition-opacity duration-300"
                style={{
                    background: `radial-gradient(800px circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(6,182,212,0.1), transparent 40%)`
                }}
            />

            {/* 2. Animated Ambient Background Blobs & Light Leaks */}
            <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-cyan-600/15 blur-[150px] rounded-full mix-blend-screen animate-pulse duration-10000"></div>
                <div className="absolute bottom-[-20%] right-[-10%] w-[40%] h-[40%] bg-blue-700/15 blur-[150px] rounded-full mix-blend-screen animate-pulse duration-7000"></div>

                {/* Additional Light Leaks */}
                <div className="absolute top-[30%] right-[-15%] w-[40%] h-[40%] bg-purple-600/10 blur-[150px] rounded-full mix-blend-screen animate-pulse shadow-[0_0_100px_rgba(147,51,234,0.1)]" style={{ animationDelay: '2s', animationDuration: '8s' }}></div>
                <div className="absolute top-[20%] left-[30%] w-[35%] h-[35%] bg-emerald-500/10 blur-[130px] rounded-full mix-blend-screen animate-pulse" style={{ animationDelay: '4s', animationDuration: '11s' }}></div>
                <div className="absolute bottom-[5%] left-[-10%] w-[35%] h-[35%] bg-pink-500/10 blur-[150px] rounded-full mix-blend-screen animate-pulse" style={{ animationDelay: '1s', animationDuration: '9s' }}></div>
            </div>

            {/* Main Content Container (z-10 puts it ABOVE the glow) */}
            <div className="max-w-7xl mx-auto p-8 relative z-10 pt-16">
                <header className="mb-16 text-center">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-cyan-400 font-bold text-xs tracking-[0.2em] uppercase mb-6 shadow-[0_0_30px_rgba(6,182,212,0.15)] backdrop-blur-md">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                        System Active
                    </div>
                    <h1 className="text-6xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-100 to-blue-400 mb-4 drop-shadow-2xl">
                        Clinical Vision AI
                    </h1>
                    <p className="text-slate-400 text-lg max-w-2xl mx-auto font-light">
                        Multi-modal neural fusion with Grad-CAM++ attribution. Upload a plantar
                        scan for explainable diagnostic triage.
                    </p>

                    {/* A tool that outputs a risk band next to the word "clinical" needs
                        its status stated on the page, not only in the exported PDF. */}
                    <div className="mt-6 mx-auto max-w-3xl flex items-start gap-3 px-5 py-3 rounded-2xl bg-amber-500/10 border border-amber-500/25 text-left">
                        <svg className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M5.07 19h13.86a2 2 0 001.71-3.03l-6.93-12a2 2 0 00-3.42 0l-6.93 12A2 2 0 005.07 19z" />
                        </svg>
                        <p className="text-amber-200/90 text-xs leading-relaxed">
                            <strong className="text-amber-300">Research prototype — not a medical device.</strong>{" "}
                            Not FDA/CE cleared and not validated against clinical outcomes. The
                            clinical risk coefficients are literature-informed priors, not fitted
                            to patient data. Never use this in place of assessment by a qualified
                            clinician.
                        </p>
                    </div>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                    <div className="lg:col-span-5">
                        <UploadForm setResult={setResult} setLoading={setLoading} loading={loading} />
                    </div>
                    <div className="lg:col-span-7">
                        <ResultCard result={result} loading={loading} />
                    </div>
                </div>
            </div>
        </main>
    );
}