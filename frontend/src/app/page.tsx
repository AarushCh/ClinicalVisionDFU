"use client";
import { useState } from "react";
import UploadForm from "@/components/UploadForm";
import ResultCard from "@/components/ResultCard";

export default function Home() {
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    return (
        <main className="min-h-screen bg-[#0B1120] p-8 text-slate-200 font-sans selection:bg-blue-500/30">
            <div className="max-w-6xl mx-auto">
                <header className="mb-12 text-center">
                    <div className="inline-block p-2 px-4 rounded-full bg-blue-500/10 text-blue-400 font-medium text-sm mb-4 border border-blue-500/20">
                        Nexus Neural Diagnostics
                    </div>
                    <h1 className="text-5xl font-extrabold tracking-tight text-white mb-2">
                        Explainable AI Clinical Support
                    </h1>
                    <p className="text-slate-400 mt-2 text-lg max-w-2xl mx-auto">
                        Advanced multi-modal fusion engine for high-precision Diabetic Foot Ulcer detection and interpretability.
                    </p>
                </header>
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
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