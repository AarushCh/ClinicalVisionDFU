"use client";
import { useState } from "react";
import UploadForm from "@/components/UploadForm";
import ResultCard from "@/components/ResultCard";

export default function Home() {
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    return (
        <main className="min-h-screen bg-[#030712] text-slate-200 font-sans selection:bg-cyan-500/30 relative overflow-hidden">
            {/* Animated Ambient Background Blobs */}
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-cyan-600/20 blur-[150px] rounded-full mix-blend-screen animate-pulse duration-10000"></div>
                <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-blue-700/20 blur-[150px] rounded-full mix-blend-screen animate-pulse duration-7000"></div>
            </div>

            <div className="max-w-7xl mx-auto p-8 relative z-10 pt-16">
                <header className="mb-16 text-center">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-cyan-400 font-bold text-xs tracking-[0.2em] uppercase mb-6 shadow-[0_0_30px_rgba(6,182,212,0.15)] backdrop-blur-md">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                        System Active
                    </div>
                    <h1 className="text-6xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-100 to-blue-400 mb-4 drop-shadow-2xl">
                        ClinicalVision AI
                    </h1>
                    <p className="text-slate-400 text-lg max-w-2xl mx-auto font-light">
                        Next-generation multi-modal neural fusion. Upload plantar scans for immediate Explainable AI diagnostic triage.
                    </p>
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