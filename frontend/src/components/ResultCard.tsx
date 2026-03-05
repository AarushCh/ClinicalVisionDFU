"use client";

interface ResultCardProps {
    result: any;
    loading: boolean;
}

export default function ResultCard({ result, loading }: ResultCardProps) {
    if (loading) {
        return (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 h-full flex flex-col items-center justify-center min-h-[300px]">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                <p className="text-blue-600 font-medium animate-pulse">Generating Map...</p>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 h-full flex flex-col items-center justify-center text-slate-400 border-dashed border-2 min-h-[300px]">
                <p>Upload data to view analysis.</p>
            </div>
        );
    }

    const riskColors: Record<string, string> = {
        HIGH: "bg-red-100 text-red-800 border-red-200",
        MEDIUM: "bg-yellow-100 text-yellow-800 border-yellow-200",
        LOW: "bg-green-100 text-green-800 border-green-200"
    };

    const activeColor = riskColors[result.risk] || riskColors.LOW;

    return (
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
            <h2 className="text-xl font-bold mb-4">Diagnostic Result</h2>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-slate-500 uppercase">Risk Level</span>
                    <span className={`px-4 py-1.5 rounded-full font-bold text-sm border ${activeColor}`}>
                        {result.risk} RISK ({(result.confidence * 100).toFixed(1)}%)
                    </span>
                </div>
                <div>
                    <h3 className="text-sm font-semibold text-slate-500 mb-2 uppercase">XAI Proof</h3>
                    <img
                        src={`data:image/jpeg;base64,${result.heatmap}`}
                        alt="Heatmap"
                        className="w-full rounded-lg border shadow-sm object-cover"
                    />
                </div>
                <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                    <h3 className="text-sm font-bold mb-1">Justification</h3>
                    <p className="text-sm text-slate-600">{result.explanation}</p>
                </div>
            </div>
        </div>
    );
}