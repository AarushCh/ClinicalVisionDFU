"use client";

export default function ResultCard({ result, loading }: any) {
    if (loading) {
        return (
            <div className="bg-[#1e293b]/50 backdrop-blur-xl p-8 rounded-3xl border border-white/10 shadow-2xl h-full flex flex-col items-center justify-center min-h-[500px]">
                <div className="relative w-24 h-24">
                    <div className="absolute inset-0 border-t-4 border-blue-500 rounded-full animate-spin"></div>
                    <div className="absolute inset-2 border-r-4 border-indigo-400 rounded-full animate-spin reverse"></div>
                </div>
                <p className="mt-6 text-blue-400 font-medium animate-pulse tracking-widest uppercase text-sm">Extracting Gradients...</p>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="bg-[#0f172a] p-8 rounded-3xl border border-dashed border-slate-700 h-full flex flex-col items-center justify-center min-h-[500px] text-slate-500">
                <svg className="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                <p className="text-lg">System awaiting multi-modal input.</p>
            </div>
        );
    }

    const riskBadge = {
        HIGH: { bg: "bg-rose-500/20", border: "border-rose-500/50", text: "text-rose-400", glow: "shadow-[0_0_30px_rgba(244,63,94,0.3)]" },
        MEDIUM: { bg: "bg-amber-500/20", border: "border-amber-500/50", text: "text-amber-400", glow: "shadow-[0_0_30px_rgba(245,158,11,0.3)]" },
        LOW: { bg: "bg-emerald-500/20", border: "border-emerald-500/50", text: "text-emerald-400", glow: "shadow-[0_0_30px_rgba(16,185,129,0.3)]" }
    }[result.risk as string] || { bg: "bg-slate-800", border: "border-slate-600", text: "text-slate-400", glow: "" };

    return (
        <div className="bg-[#1e293b]/50 backdrop-blur-xl p-8 rounded-3xl border border-white/10 shadow-2xl h-full flex flex-col">

            {/* Risk Header */}
            <div className="flex items-center justify-between mb-8 pb-6 border-b border-slate-700/50">
                <div>
                    <h2 className="text-2xl font-bold text-white">Diagnostic Output</h2>
                    <p className="text-slate-400 text-sm mt-1">Confidence Score: {(result.confidence * 100).toFixed(2)}%</p>
                </div>
                <div className={`px-6 py-2 rounded-full font-black tracking-widest border-2 ${riskBadge.bg} ${riskBadge.border} ${riskBadge.text} ${riskBadge.glow}`}>
                    {result.risk} RISK
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 flex-1">

                {/* Visual Proof */}
                <div className="flex flex-col">
                    <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
                        Grad-CAM Heatmap
                    </h3>
                    <div className="relative rounded-2xl overflow-hidden border border-slate-700 shadow-lg flex-1 group">
                        <img src={`data:image/jpeg;base64,${result.heatmap}`} alt="Heatmap" className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                            <p className="text-xs text-slate-300 font-medium">Red zones indicate CNN attention focus</p>
                        </div>
                    </div>
                </div>

                {/* SHAP & Justification */}
                <div className="flex flex-col gap-6">

                    {/* SHAP Feature Importance */}
                    <div>
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">SHAP Feature Importance</h3>
                        <div className="space-y-4">
                            {result.shap?.map((s: any, i: number) => (
                                <div key={i}>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-slate-300 font-medium">{s.feature}</span>
                                        <span className="text-blue-400 font-bold">{s.value}%</span>
                                    </div>
                                    <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-blue-500 to-indigo-400 rounded-full transition-all duration-1000"
                                            style={{ width: `${s.value}%` }}
                                        ></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Clinical Justification */}
                    <div className="bg-[#0f172a] p-5 rounded-2xl border border-slate-700 mt-auto">
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Model Justification</h3>
                        <p className="text-slate-300 text-sm leading-relaxed">{result.explanation}</p>
                    </div>

                </div>
            </div>
        </div>
    );
}