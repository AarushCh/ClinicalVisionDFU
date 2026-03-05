"use client";

import { useState } from "react";
import UploadForm from "@/components/UploadForm";
import ResultCard from "@/components/ResultCard";

export default function Home() {
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    return (
        <main className="min-h-screen p-8">
            <div className="max-w-6xl mx-auto">
                <header className="mb-10 text-center">
                    <h1 className="text-4xl font-extrabold tracking-tight">
                        Explainable AI Clinical Support
                    </h1>
                    <p className="text-slate-500 mt-2 text-lg">Diabetic Foot Ulcer Detection</p>
                </header>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <UploadForm setResult={setResult} setLoading={setLoading} loading={loading} />
                    <ResultCard result={result} loading={loading} />
                </div>
            </div>
        </main>
    );
}