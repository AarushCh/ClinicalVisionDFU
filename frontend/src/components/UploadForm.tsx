"use client";
import { useState } from "react";
import axios from "axios";

export default function UploadForm({ setResult, setLoading, loading }: any) {
    const [file, setFile] = useState<File | null>(null);
    const [formData, setFormData] = useState({ age: "", bmi: "", diabetes_years: "" });

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) setFile(e.target.files[0]);
    };
    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const onSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) return;
        setLoading(true);
        const data = new FormData();
        data.append("image", file);
        data.append("age", formData.age);
        data.append("bmi", formData.bmi);
        data.append("diabetes_years", formData.diabetes_years);

        try {
            const res = await axios.post("http://127.0.0.1:10000/predict", data);
            setResult(res.data);
        } catch (error) {
            alert("Backend connection failed.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white/[0.02] backdrop-blur-2xl p-8 rounded-[2rem] border border-white/10 shadow-[0_8px_32px_0_rgba(0,0,0,0.36)] relative group">
            <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-[2rem] pointer-events-none"></div>

            <h2 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
                <div className="p-2 bg-cyan-500/20 rounded-lg border border-cyan-500/30 text-cyan-400">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                </div>
                Patient Parameters
            </h2>

            <form onSubmit={onSubmit} className="space-y-6 relative z-10">

                <div className="relative group/drop cursor-pointer">
                    <div className="absolute inset-0 bg-cyan-500/20 rounded-2xl blur-xl group-hover/drop:bg-cyan-500/30 transition-all duration-500"></div>
                    <div className="relative border-2 border-dashed border-white/20 rounded-2xl p-8 text-center bg-black/40 hover:bg-black/60 hover:border-cyan-400/50 transition-all duration-300 backdrop-blur-sm overflow-hidden">
                        <input type="file" onChange={handleFileChange} accept="image/*" className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20" required />
                        <div className="relative z-10">
                            <div className="w-16 h-16 mx-auto bg-white/5 rounded-full flex items-center justify-center mb-4 border border-white/10 group-hover/drop:scale-110 transition-transform">
                                <svg className="h-8 w-8 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                            </div>
                            <p className="text-sm font-bold text-white mb-1">{file ? file.name : "Initialize Image Upload"}</p>
                            <p className="text-xs text-slate-400">{file ? "Ready for extraction" : "Drag and drop planar scan here"}</p>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-5">
                    <div className="space-y-2">
                        <label className="text-[10px] font-black text-cyan-500 uppercase tracking-[0.15em]">Patient Age</label>
                        <input type="number" name="age" onChange={handleInputChange} className="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-white focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 outline-none transition-all placeholder-slate-600 font-medium" required placeholder="e.g. 55" />
                    </div>
                    <div className="space-y-2">
                        <label className="text-[10px] font-black text-cyan-500 uppercase tracking-[0.15em]">BMI Index</label>
                        <input type="number" step="0.1" name="bmi" onChange={handleInputChange} className="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-white focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 outline-none transition-all placeholder-slate-600 font-medium" required placeholder="e.g. 28.5" />
                    </div>
                </div>

                <div className="space-y-2">
                    <label className="text-[10px] font-black text-cyan-500 uppercase tracking-[0.15em]">Diabetes Duration (Years)</label>
                    <input type="number" name="diabetes_years" onChange={handleInputChange} className="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-white focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 outline-none transition-all placeholder-slate-600 font-medium" required placeholder="e.g. 10" />
                </div>

                <button type="submit" disabled={loading} className="w-full relative group/btn mt-8 overflow-hidden rounded-xl p-[1px]">
                    <span className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-blue-500 to-cyan-500 animate-[spin_3s_linear_infinite] group-hover/btn:opacity-100 opacity-70"></span>
                    <div className="relative bg-black/80 backdrop-blur-xl px-8 py-5 rounded-xl transition-all group-hover/btn:bg-transparent">
                        <span className="text-white font-bold tracking-wider text-sm flex items-center justify-center gap-2">
                            {loading ? (
                                <>
                                    <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                    EXECUTING NEURAL FUSION...
                                </>
                            ) : "ENGAGE AI ANALYSIS"}
                        </span>
                    </div>
                </button>
            </form>
        </div>
    );
}